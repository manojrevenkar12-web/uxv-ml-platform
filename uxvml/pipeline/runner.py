"""Local DAG runner with content-keyed step caching.

The same steps run as containers under Argo Workflows (deploy/argo). Locally,
each step's cache key is a hash of its parameters and the identities of its
inputs (dataset version, model weights). If a step with the same key has
already succeeded, its recorded outputs are reused instead of recomputing.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

from ..data.dataset import IngestParams, build_dataset, load_index, load_manifest
from ..data.store import open_store


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def calibration_tags(truth_dir: Path, flights: list[str]):
    """Stands in for human tagging of a few flights (lighting, fog) in the simulator."""
    import pandas as pd

    fog_name = {0.0: "none", 0.3: "light", 0.55: "heavy"}
    rows = []
    for f in flights:
        t = json.loads((Path(truth_dir) / f"{f}.json").read_text())
        for fr in t["frames"]:
            rows.append({"sample_id": f"{f}/{fr['ts_us']}", "lighting": t["spec"]["tod"],
                         "fog": fog_name[t["spec"]["fog"]]})
    return pd.DataFrame(rows)


class Pipeline:
    def __init__(self, config_path: str | Path, force: bool = False, log=print):
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        self.base = Path(config_path).resolve().parent.parent
        self.work = (self.base / self.cfg["workdir"]).resolve()
        self.store_uri = str(self.work / "store")
        self.tracking = f"sqlite:///{self.work / 'mlflow.db'}"
        self.store = open_store(self.store_uri)
        self.force, self.log = force, log
        self.timings = {}

    def _cached(self, step: str, key: str, fn):
        ref = f"pipeline/cache/{step}/{key}"
        if not self.force and self.store.has_ref(ref):
            self.log(f"[{step}] cache hit ({key})")
            self.timings[step] = 0.0
            return self.store.get_ref(ref)
        t = time.time()
        self.log(f"[{step}] running ({key})")
        out = fn()
        self.timings[step] = round(time.time() - t, 1)
        self.store.put_ref(ref, out)
        return out

    def run(self) -> dict:
        c = self.cfg
        sim = c["simulate"]
        raw = self.work / "sim" / "raw"

        def do_sim():
            from ..sim import simulate
            simulate(self.work / "sim", sim["flights"], sim["seed"], sim["duration_s"], c.get("workers", 2))
            return {"raw": str(raw)}
        self._cached("simulate", _key(sim), do_sim)

        ing = c["ingest"]
        params = IngestParams(**ing.get("params", {}))

        def do_ingest():
            m = build_dataset(raw, self.store, ing["dataset"], params, c.get("workers", 2), log=self.log)
            return {"dataset": ing["dataset"], "version": m["version"]}
        ds = self._cached("ingest", _key(sim, ing), do_ingest)

        tr = c["train"]

        def do_train():
            from ..train.trainer import TrainConfig, train
            cfg = TrainConfig(dataset=ds["dataset"], dataset_version=ds["version"], **tr)
            out = self.work / "models" / f"{tr.get('run_name', 'train')}-{ds['version'][:8]}"
            card = train(cfg, self.store_uri, self.tracking, str(out))
            return {"model_dir": str(out), "weights": card["weights_sha256"], "val_mAP": card["val_mAP"]}
        model = self._cached("train", _key(ds, tr), do_train)

        manifest = load_manifest(self.store, ds["dataset"], ds["version"])

        def do_enrich():
            from ..curate.enrich import enrich, save_enrichment
            from ..train.trainer import load_model
            index = load_index(self.store, manifest)
            val_flights = sorted(index[index.split == "val"].flight_id.unique())
            tags = calibration_tags(self.work / "sim" / "truth", val_flights)
            m, card = load_model(model["model_dir"])
            table, emb, order = enrich(self.store, manifest, m, card["config"]["inputs"], tags)
            table = table.merge(index.drop(columns=["split"]), on="sample_id")
            doc = save_enrichment(self.store, manifest, table, emb, order, model["weights"])
            return doc
        self._cached("enrich", _key(ds, model["weights"], "enrich-v1"), do_enrich)

        val = c["validate"]

        def do_validate():
            from ..curate.enrich import load_enrichment
            from ..validate.gate import GateConfig, compare, promote, render_markdown
            gcfg = GateConfig.from_yaml(self.base / val["gates"])
            ref = f"models/{val['model_name']}/production"
            baseline = self.store.get_ref(ref)["model_dir"] if self.store.has_ref(ref) else None
            if baseline == model["model_dir"]:
                baseline = None
            table, _, _ = load_enrichment(self.store, manifest)
            report = compare(model["model_dir"], baseline, self.store, manifest, gcfg, table)
            out = Path(model["model_dir"])
            (out / "validation.json").write_text(json.dumps(report, indent=2, default=float))
            (out / "validation.md").write_text(render_markdown(report))
            if report["decision"] == "promote":
                promote(model["model_dir"], self.tracking, val["model_name"], self.store)
            return {"decision": report["decision"], "report": str(out / "validation.md")}
        result = self._cached("validate", _key(ds, model["weights"], val), do_validate)
        return {"dataset": ds, "model": model, "validation": result, "timings": self.timings}
