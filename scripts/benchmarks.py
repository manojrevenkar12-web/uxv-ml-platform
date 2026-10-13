"""Scores every platform component against the simulator's hidden truth and
writes WORKDIR/results/benchmarks.json (summarised in docs/RESULTS.md).

Usage: python scripts/benchmarks.py WORKDIR [--only models]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uxvml.curate.autolabel import audit_labels
from uxvml.curate.enrich import enrich, load_enrichment, save_enrichment
from uxvml.curate.search import parse_query, run_query, similar
from uxvml.data.dataset import load_index, load_manifest
from uxvml.data.loader import ShardStream
from uxvml.data.raw import load_flight
from uxvml.data.sync import fusion_raster
from uxvml.pipeline.runner import calibration_tags
from uxvml.sim.world import CLASSES
from uxvml.train.targets import map_from_records, match_tolerance, per_class_records
from uxvml.train.trainer import load_model, predict
from uxvml.validate.gate import GateConfig, compare, render_markdown

FOG = {0.0: "none", 0.3: "light", 0.55: "heavy"}


def truth_tables(sim: Path):
    specs, frames = {}, []
    for p in sorted((sim / "truth").glob("*.json")):
        t = json.loads(p.read_text())
        s = t["spec"]
        specs[s["flight_id"]] = t
        for fr in t["frames"]:
            cnt = np.bincount([b["cls"] for b in fr["boxes"]], minlength=3) if fr["boxes"] else np.zeros(3, int)
            frames.append({"sample_id": f"{s['flight_id']}/{fr['ts_us']}", "true_tod": s["tod"],
                           "true_fog": FOG[s["fog"]], **{f"true_{c}": int(cnt[i]) for i, c in enumerate(CLASSES)}})
    return specs, pd.DataFrame(frames)


def qc_bench(store, manifest, specs):
    qc = json.loads(store.get(manifest["qc_report"]))
    out = {}
    inj = {(f, str(c["ts_us"])) for f, t in specs.items() for c in t["camera_faults"]}
    flag = {(f, ts) for f, q in qc.items() for ts in q["frame_faults"]}
    out["camera_faults"] = {"injected": len(inj), "flagged": len(flag), "true_positive": len(inj & flag)}

    rows = []
    for f, t in specs.items():
        s = qc[f]["sync"]
        rows.append({"flight": f, "true_offset_s": t["spec"]["lidar_clock_offset_s"], "status": s["status"],
                     "estimated_s": s.get("offset_s"), "applied_s": s["applied_offset_s"],
                     "tod": t["spec"]["tod"], "fog": t["spec"]["fog"]})
    df = pd.DataFrame(rows)
    off = df[df.true_offset_s != 0]
    corrected = off[off.status == "corrected"]
    out["sync"] = {
        "flights": len(df), "flights_with_injected_offset": len(off),
        "corrected": len(corrected),
        "max_abs_error_ms_when_corrected": float((corrected.applied_s - corrected.true_offset_s).abs().max() * 1000) if len(corrected) else None,
        "wrong_corrections": int(((df.status == "corrected") & ((df.applied_s - df.true_offset_s).abs() > 0.03)).sum()),
        "unverified_flights": int((df.status == "unverified").sum()),
        "unverified_with_offset": int((off.status == "unverified").sum()),
        "detail": rows,
    }
    gap_truth = {f for f, t in specs.items() if t["spec"]["lidar_gap"]}
    gap_flag = {f for f, q in qc.items() if "lidar_gap_over_1s" in q["issues"]}
    out["lidar_gaps"] = {"injected": len(gap_truth), "flagged": len(gap_flag), "true_positive": len(gap_truth & gap_flag)}

    # physical-size rule vs injected annotation errors
    n_wrong = n_wrong_hit = n_flags = n_flags_real = 0
    for f, t in specs.items():
        if not t["spec"]["noisy_annotations"]:
            continue
        errs = {}
        for e in t["annotation_errors"]:
            errs[(e["ts_us"], e["obj"])] = e["error"]
        suspects = qc[f]["suspect_boxes"]
        for fr in t["frames"]:
            ts = fr["ts_us"]
            objs = [b["obj"] for b in fr["boxes"] if errs.get((ts, b["obj"])) != "missing"]
            flagged = set(suspects.get(str(ts), []))
            for j, o in enumerate(objs):
                e = errs.get((ts, o))
                if e == "wrong_class":
                    n_wrong += 1
                    n_wrong_hit += j in flagged
            n_flags += len(flagged)
            n_flags_real += sum(1 for j in flagged if j < len(objs) and (ts, objs[j]) in errs)
    out["size_rule"] = {"wrong_class_errors": n_wrong, "caught": n_wrong_hit,
                        "flags_on_noisy_flights": n_flags, "flags_that_are_real_errors": n_flags_real}
    return out


def throughput_bench(store, manifest, raw_root: Path, n=400):
    idx = load_index(store, manifest)
    sub = idx[idx.split == "val"].head(n)
    flights = {}
    t = time.time()
    for _, r in sub.iterrows():
        f = flights.setdefault(r.flight_id, load_flight(raw_root / r.flight_id))
        f.image(int(r.ts_us))
        fusion_raster(f, int(r.ts_us), 0.0)
    raw_rate = len(sub) / (time.time() - t)

    tmp = Path(tempfile.mkdtemp())
    items = list(ShardStream(store, manifest, "val", train=False))[:n]
    for i, it in enumerate(items):
        np.save(tmp / f"{i}_x.npy", it["x"].numpy())
        np.save(tmp / f"{i}_b.npy", it["boxes"])
    t = time.time()
    for i in range(len(items)):
        np.load(tmp / f"{i}_x.npy")
        np.load(tmp / f"{i}_b.npy")
    file_rate = len(items) / (time.time() - t)

    t = time.time()
    cnt = 0
    for _ in ShardStream(store, manifest, "val", train=False):
        cnt += 1
    shard_rate = cnt / (time.time() - t)
    return {"raw_decode_and_project_sps": raw_rate, "one_file_per_sample_sps": file_rate,
            "shard_stream_sps": shard_rate, "note": "single process, warm OS page cache, local disk"}


def eval_models(store, manifest, models: dict, truth: pd.DataFrame):
    res = {}
    for name, d in models.items():
        if not (Path(d) / "model_card.json").exists():
            continue
        m, card = load_model(d)
        sids, preds, gts = predict(m, store, manifest, "test", card["config"]["inputs"])
        recs = per_class_records(preds, gts)
        aps, mp = map_from_records(recs)
        tod = truth.set_index("sample_id").loc[sids].true_tod.values
        per_tod = {}
        for v in ("day", "dusk", "night"):
            rows = np.nonzero(tod == v)[0]
            per_tod[v] = map_from_records([recs[i] for i in rows])[1]
        res[name] = {"test_mAP": mp, **{f"AP_{CLASSES[c]}": a for c, a in aps.items()},
                     **{f"mAP_{k}": v for k, v in per_tod.items()}, "weights": card["weights_sha256"][:16],
                     "dataset_version": card["dataset_version"], "train_seconds": None}
    return res


def enrichment_bench(store, manifest, model_dir, sim, truth):
    idx = load_index(store, manifest)
    val_flights = sorted(idx[idx.split == "val"].flight_id.unique())
    tags = calibration_tags(sim / "truth", val_flights)
    m, card = load_model(model_dir)
    table, emb, order = enrich(store, manifest, m, card["config"]["inputs"], tags)
    table = table.merge(idx.drop(columns=["split"]), on="sample_id")
    save_enrichment(store, manifest, table, emb, order, card["weights_sha256"])
    t = table.merge(truth, on="sample_id")
    held = t[~t.flight_id.isin(val_flights)]
    out = {"calibration_flights": len(val_flights), "evaluated_frames": len(held),
           "lighting_accuracy": float((held.lighting == held.true_tod).mean()),
           "fog_accuracy": float((held.fog == held.true_fog).mean())}
    trusted = held[held.vendor == "vendor-b"]
    for c in CLASSES:
        out[f"count_mae_{c}"] = float((trusted[f"pred_{c}"] - trusted[f"true_{c}"]).abs().mean())
    return out


SEARCH_QUERIES = [
    ("night frames with at least 3 vehicles", lambda r: r.true_tod == "night" and r.true_vehicle >= 3),
    ("foggy dusk frames with people", lambda r: r.true_tod == "dusk" and r.true_fog != "none" and r.true_person >= 1),
    ("daytime frames with at least 2 buildings below 35 m",
     lambda r: r.true_tod == "day" and r.true_building >= 2 and r.altitude_m <= 35),
    ("heavy fog with more than 2 vehicles", lambda r: r.true_fog == "heavy" and r.true_vehicle >= 3),
    ("clear night frames with people", lambda r: r.true_tod == "night" and r.true_fog == "none" and r.true_person >= 1),
    ("crowded frames", lambda r: r.true_person >= 5),
]


def search_bench(store, manifest, truth):
    table, emb, order = load_enrichment(store, manifest)
    t = table.merge(truth, on="sample_id")
    out = []
    for q, ok in SEARCH_QUERIES:
        fq = parse_query(q)
        fq.limit = 10000
        res = run_query(t, fq)
        hits = [bool(ok(r)) for r in res.itertuples()]
        relevant = sum(bool(ok(r)) for r in t.itertuples())
        out.append({"query": q, "returned": len(res), "precision": float(np.mean(hits)) if hits else None,
                    "recall": (sum(hits) / relevant) if relevant else None, "relevant_in_dataset": relevant})
    # query-by-example: do neighbours share the query frame's conditions?
    cond = t.set_index("sample_id")[["true_tod", "true_fog"]].apply(tuple, axis=1).to_dict()
    rng = np.random.default_rng(0)
    qs = rng.choice(order, 100, replace=False)
    p_at_10 = np.mean([np.mean([cond[s] == cond[q] for s, _ in similar(emb, order, q, 10)]) for q in qs])
    base = np.mean([np.mean([cond[s] == cond[q] for s in rng.choice(order, 10)]) for q in qs])
    return {"text_queries": out, "similar_same_condition_p_at_10": float(p_at_10),
            "random_same_condition_p_at_10": float(base)}


def audit_bench(store, manifest, model_dir, specs):
    idx = load_index(store, manifest)
    noisy = set(idx[(idx.split == "train") & (idx.vendor == "vendor-a")].sample_id)
    m, card = load_model(model_dir)
    flags = audit_labels(m, store, manifest, card["config"]["inputs"], noisy)
    by_sid = {}
    for f in flags:
        by_sid.setdefault(f["sample_id"], []).append(f)
    n_err = {"missing": 0, "shifted": 0, "wrong_class": 0}
    caught = {"missing": 0, "shifted": 0, "wrong_class": 0}
    real_flags = 0
    total_boxes = 0
    for sid in noisy:
        fid, ts = sid.split("/")
        t = specs[fid]
        errs = {(e["ts_us"], e["obj"]): e["error"] for e in t["annotation_errors"] if e["ts_us"] == int(ts)}
        fr = next(x for x in t["frames"] if x["ts_us"] == int(ts))
        clean = fr["boxes"]
        ann_objs = [b["obj"] for b in clean if errs.get((int(ts), b["obj"])) != "missing"]
        total_boxes += len(ann_objs)
        hit_objs = set()
        for f in by_sid.get(sid, []):
            obj = None
            if f["box"] is not None:
                obj = ann_objs[f["box"]]
            else:  # predicted object without a label: which true object is there?
                px, py = f["pred"][1], f["pred"][2]
                for b in clean:
                    if np.hypot(b["x"] - px, b["y"] - py) <= max(match_tolerance(b["w"], b["h"]), 0.5 * max(b["w"], b["h"])):
                        obj = b["obj"]
                        break
            if obj is not None and (int(ts), obj) in errs:
                real_flags += 1
                hit_objs.add(obj)
        for (ts2, obj), e in errs.items():
            n_err[e] += 1
            caught[e] += obj in hit_objs
    return {"frames_audited": len(noisy), "boxes": total_boxes, "flags": len(flags),
            "flags_pointing_at_real_errors": real_flags,
            "errors": n_err, "caught": caught}


def main():
    work = Path(sys.argv[1])
    sim = work / "sim"
    from uxvml.data.store import open_store
    store = open_store(work / "store")
    manifest = load_manifest(store, "uxv")
    specs, truth = truth_tables(sim)
    results = {"dataset": {"version": manifest["version"], **manifest["stats"]}}
    only_models = "--only" in sys.argv
    if not only_models:
        print("qc...", flush=True)
        results["qc"] = qc_bench(store, manifest, specs)
        print("throughput...", flush=True)
        results["throughput"] = throughput_bench(store, manifest, sim / "raw")
    models = {n: work / "models" / n for n in ["fusion", "camera-only", "fusion-repro", "fusion-nosync",
                                                "fusion-trusted-labels", "autolabel-m0", "autolabel-m1",
                                                "fusion-seed1", "fusion-seed2"]}
    print("models...", flush=True)
    results["models"] = eval_models(store, manifest, models, truth)
    seeds = [results["models"][n]["test_mAP"] for n in ("fusion", "fusion-seed1", "fusion-seed2")
             if n in results["models"]]
    if len(seeds) > 1:
        results["seed_spread"] = {"test_mAP": seeds, "min": min(seeds), "max": max(seeds),
                                  "range": max(seeds) - min(seeds), "std": float(np.std(seeds, ddof=1))}
    if "--only" in sys.argv and sys.argv[sys.argv.index("--only") + 1] == "models":
        out = work / "results" / "benchmarks.json"
        merged = json.loads(out.read_text())
        merged["models"] = results["models"]
        if "seed_spread" in results:
            merged["seed_spread"] = results["seed_spread"]
        out.write_text(json.dumps(merged, indent=2, default=float))
        print(json.dumps({"models": results["models"], "seed_spread": results.get("seed_spread")}, indent=1, default=float))
        return
    print("enrichment...", flush=True)
    results["enrichment"] = enrichment_bench(store, manifest, models["fusion"], sim, truth)
    print("search...", flush=True)
    results["search"] = search_bench(store, manifest, truth)
    print("audit...", flush=True)
    results["label_audit"] = audit_bench(store, manifest, models["fusion-trusted-labels"], specs)
    print("gates...", flush=True)
    table, _, _ = load_enrichment(store, manifest)
    gates = {}
    cfg = GateConfig.from_yaml(Path(__file__).parent.parent / "configs" / "gates.yaml")
    for cand, base in [("fusion", "camera-only"), ("camera-only", "fusion"), ("fusion-repro", "fusion"),
                       ("fusion-nosync", "fusion"), ("fusion", "fusion-trusted-labels")]:
        if (models[cand] / "model_card.json").exists() and (models[base] / "model_card.json").exists():
            rep = compare(models[cand], models[base], store, manifest, cfg, table)
            (work / "results").mkdir(exist_ok=True)
            (work / "results" / f"gate_{cand}_vs_{base}.md").write_text(render_markdown(rep))
            gates[f"{cand} vs {base}"] = {"decision": rep["decision"],
                                         "failed": [c["check"] for c in rep["checks"] if not c["passed"]],
                                         "mAP_delta": rep["deltas"]["mAP"]}
    results["gates"] = gates
    al = work / "results" / "autolabel.json"
    if al.exists():
        results["autolabel"] = json.loads(al.read_text())
    (work / "results").mkdir(exist_ok=True)
    (work / "results" / "benchmarks.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: v for k, v in results.items() if k != "qc"}, indent=1, default=float))


if __name__ == "__main__":
    main()
