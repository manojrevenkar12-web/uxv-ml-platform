"""Command-line entry points. Each subcommand is one pipeline step, so the same
commands run locally, in CI, and as Argo Workflow steps on Kubernetes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_simulate(a):
    from .sim import simulate
    specs = simulate(a.out, a.flights, a.seed, a.duration, a.workers)
    _print({"flights": len(specs), "raw": str(Path(a.out) / "raw")})


def cmd_ingest(a):
    from .data.dataset import IngestParams, build_dataset
    from .data.store import open_store
    m = build_dataset(Path(a.raw), open_store(a.store), a.dataset,
                      IngestParams(sync_correction=not a.no_sync_correction, shard_size=a.shard_size,
                                   n_val_per_tod=a.val_per_group, n_test_per_tod=a.test_per_group),
                      a.workers)
    _print({"version": m["version"], **m["stats"]})
    if a.version_file:
        Path(a.version_file).write_text(m["version"])


def cmd_train(a):
    from .train.trainer import TrainConfig, train
    cfg = TrainConfig(dataset=a.dataset, dataset_version=a.version, inputs=a.inputs, epochs=a.epochs,
                      batch_size=a.batch_size, world_size=a.world_size, seed=a.seed,
                      run_name=a.run_name, threads_per_rank=a.threads)
    card = train(cfg, a.store, a.tracking_uri, a.out)
    if card:
        _print({k: card[k] for k in ("dataset_version", "weights_sha256", "val_mAP", "mlflow_run_id")})


def cmd_enrich(a):
    import pandas as pd

    from .curate.enrich import enrich, save_enrichment
    from .data.dataset import load_index, load_manifest
    from .data.store import open_store
    from .train.trainer import load_model
    store = open_store(a.store)
    manifest = load_manifest(store, a.dataset, a.version)
    model, card = load_model(a.model)
    index = load_index(store, manifest)
    tags = pd.read_csv(a.calibration) if a.calibration else None
    if a.truth:  # simulator: use the validation flights as the hand-tagged calibration set
        from .pipeline.runner import calibration_tags
        tags = calibration_tags(Path(a.truth), sorted(index[index.split == "val"].flight_id.unique()))
    table, emb, order = enrich(store, manifest, model, card["config"]["inputs"], tags)
    table = table.merge(index.drop(columns=["split"]), on="sample_id")
    _print(save_enrichment(store, manifest, table, emb, order, card["weights_sha256"]))


def cmd_tags(a):
    from .pipeline.runner import calibration_tags
    calibration_tags(Path(a.truth), a.flights.split(",")).to_csv(a.out, index=False)


def cmd_validate(a):
    from .curate.enrich import load_enrichment
    from .data.dataset import load_manifest
    from .data.store import open_store
    from .validate.gate import GateConfig, compare, promote, render_markdown
    store = open_store(a.store)
    manifest = load_manifest(store, a.dataset, a.version)
    baseline = a.baseline
    ref = f"models/{a.model_name}/production"
    if baseline is None and store.has_ref(ref):
        baseline = store.get_ref(ref)["model_dir"]
    try:
        table, _, _ = load_enrichment(store, manifest)
    except Exception:  # no enrichment yet: slices fall back to raw brightness
        table = None
    report = compare(a.candidate, baseline, store, manifest, GateConfig.from_yaml(a.gates), table)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2, default=float))
    md = render_markdown(report)
    Path(a.report).with_suffix(".md").write_text(md)
    print(md)
    if report["decision"] == "promote" and a.promote:
        promote(a.candidate, a.tracking_uri, a.model_name, store)
        print(f"promoted to {a.model_name}@production")
    if report["decision"] != "promote":
        sys.exit(3)


def cmd_search(a):
    from .curate.enrich import load_enrichment
    from .curate.search import LLMQueryParser, parse_query, run_query
    from .data.dataset import load_manifest
    from .data.store import open_store
    store = open_store(a.store)
    table, _, _ = load_enrichment(store, load_manifest(store, a.dataset, a.version))
    q = LLMQueryParser()(a.query) if a.llm else parse_query(a.query)
    res = run_query(table, q)
    print(q.model_dump_json(exclude_defaults=True))
    print(f"{len(res)} frames")
    print(res[["sample_id", "split", "lighting", "fog", "altitude_m", "pred_vehicle", "pred_person",
               "pred_building"]].to_string(index=False))


def cmd_similar(a):
    from .curate.enrich import load_enrichment
    from .curate.search import similar
    from .data.dataset import load_manifest
    from .data.store import open_store
    store = open_store(a.store)
    _, emb, order = load_enrichment(store, load_manifest(store, a.dataset, a.version))
    for sid, s in similar(emb, order, a.sample_id, a.k):
        print(f"{s:.3f}  {sid}")


def cmd_lineage(a):
    card = json.loads((Path(a.model) / "model_card.json").read_text())
    out = {"model_weights_sha256": card["weights_sha256"], "mlflow_run": card.get("mlflow_run_id"),
           "registered_version": card.get("registered_version"), "code_version": card["code_version"],
           "dataset": f"{card['dataset']}@{card['dataset_version']}", "dataset_index": card["dataset_index"],
           "raw_flights": card["raw_inputs"]}
    if a.store:
        from .data.dataset import load_manifest
        from .data.store import open_store
        m = load_manifest(open_store(a.store), card["dataset"], card["dataset_version"])
        out["shards"] = m["shards"]
        out["qc_report"] = m["qc_report"]
        out["ingest"] = {"version": m["ingest_version"], "params": m["params"]}
    _print(out)


def cmd_pipeline(a):
    from .pipeline.runner import Pipeline
    _print(Pipeline(a.config, force=a.force).run())


def main(argv=None):
    p = argparse.ArgumentParser("uxvml", description="UxV perception data & training platform")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate synthetic flight logs")
    s.add_argument("--out", required=True)
    s.add_argument("--flights", type=int, default=36)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--duration", type=float, default=32.0)
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("ingest", help="QC, sync, shard and version raw logs")
    s.add_argument("--raw", required=True)
    s.add_argument("--store", required=True)
    s.add_argument("--dataset", default="uxv")
    s.add_argument("--no-sync-correction", action="store_true")
    s.add_argument("--shard-size", type=int, default=256)
    s.add_argument("--val-per-group", type=int, default=2, help="val flights per lighting group")
    s.add_argument("--test-per-group", type=int, default=2, help="test flights per lighting group")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--version-file", help="write the dataset version id here (for workflow outputs)")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("train", help="distributed training with experiment tracking")
    s.add_argument("--store", required=True)
    s.add_argument("--tracking-uri", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dataset", default="uxv")
    s.add_argument("--version", default="latest")
    s.add_argument("--inputs", default="rgb+lidar", choices=["rgb", "rgb+lidar"])
    s.add_argument("--epochs", type=int, default=14)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--world-size", type=int, default=2)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--run-name", default="train")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("tags", help="emulate hand-tagging of calibration flights (simulator only)")
    s.add_argument("--truth", required=True)
    s.add_argument("--flights", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_tags)

    s = sub.add_parser("enrich", help="generate frame metadata + embeddings")
    s.add_argument("--store", required=True)
    s.add_argument("--dataset", default="uxv")
    s.add_argument("--version", default="latest")
    s.add_argument("--model", required=True)
    s.add_argument("--calibration", help="CSV of hand-tagged frames: sample_id,lighting,fog")
    s.add_argument("--truth", help="simulator truth dir: tag the validation flights from it")
    s.set_defaults(fn=cmd_enrich)

    s = sub.add_parser("validate", help="slice metrics + regression gate vs production")
    s.add_argument("--store", required=True)
    s.add_argument("--dataset", default="uxv")
    s.add_argument("--version", default="latest")
    s.add_argument("--candidate", required=True)
    s.add_argument("--baseline", help="model dir (default: current production ref)")
    s.add_argument("--gates", default="configs/gates.yaml")
    s.add_argument("--report", required=True)
    s.add_argument("--promote", action="store_true")
    s.add_argument("--tracking-uri", default="")
    s.add_argument("--model-name", default="uxv-detector")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("search", help="natural-language frame search")
    s.add_argument("query")
    s.add_argument("--store", required=True)
    s.add_argument("--dataset", default="uxv")
    s.add_argument("--version", default="latest")
    s.add_argument("--llm", action="store_true", help="parse with a Claude model (needs ANTHROPIC_API_KEY)")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("similar", help="query-by-example over frame embeddings")
    s.add_argument("sample_id")
    s.add_argument("--store", required=True)
    s.add_argument("--dataset", default="uxv")
    s.add_argument("--version", default="latest")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(fn=cmd_similar)

    s = sub.add_parser("lineage", help="trace a model back to raw flight logs")
    s.add_argument("--model", required=True)
    s.add_argument("--store")
    s.set_defaults(fn=cmd_lineage)

    s = sub.add_parser("pipeline", help="run the full DAG locally with step caching")
    s.add_argument("--config", default="configs/pipeline.yaml")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_pipeline)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
