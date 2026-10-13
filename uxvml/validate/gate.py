"""Model validation gate: slice metrics, paired bootstrap, promotion.

A candidate is compared with the current production model on the same frozen
test split. Differences are judged against their own sampling noise (paired
bootstrap over frames), per slice, so a candidate that is better on average
but significantly worse at night or on people is blocked.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..data.dataset import load_index, load_manifest
from ..sim.world import CLASSES
from ..train.targets import map_from_records, per_class_records
from ..train.trainer import load_model, predict


@dataclass
class GateConfig:
    min_map: float = 0.30
    overall_tolerance: float = 0.01      # allowed mAP drop vs production
    slice_tolerance: float = 0.03        # allowed per-slice / per-class drop
    n_boot: int = 400
    ci: float = 0.95
    seed: int = 0
    slices: list = field(default_factory=lambda: ["lighting", "fog", "altitude_band"])

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text()))


def slice_columns(index):
    """Slice labels derived from platform metadata (no simulator truth)."""
    df = index.copy()
    if "lighting" not in df:
        df["lighting"] = np.where(df.brightness < 30, "dark", np.where(df.brightness < 62, "dim", "bright"))
    if "fog" not in df:
        df["fog"] = "unknown"
    df["altitude_band"] = np.where(df.altitude_m < 35, "low", np.where(df.altitude_m < 48, "mid", "high"))
    return df


def records_for(model_dir, store, manifest, split="test"):
    model, card = load_model(model_dir)
    inputs = card["config"]["inputs"]
    sids, preds, gts = predict(model, store, manifest, split, inputs)
    return sids, per_class_records(preds, gts), card


def metric_set(recs, idx_rows: dict, df):
    """mAP overall, per class, and per slice value."""
    out = {}
    aps, m = map_from_records(recs)
    out["mAP"] = m
    for c, v in aps.items():
        out[f"AP/{CLASSES[c]}"] = v
    for col, groups in idx_rows.items():
        for val, rows in groups.items():
            _, ms = map_from_records([recs[i] for i in rows])
            out[f"{col}={val}/mAP"] = ms
    return out


def compare(cand_dir, base_dir, store, manifest, cfg: GateConfig, enrichment=None) -> dict:
    sids_c, rec_c, card_c = records_for(cand_dir, store, manifest)
    sids_b, rec_b, card_b = records_for(base_dir, store, manifest) if base_dir else (None, None, None)
    index = load_index(store, manifest).set_index("sample_id")
    if enrichment is not None:
        index = index.join(enrichment.set_index("sample_id")[["lighting", "fog"]], how="left")
    df = slice_columns(index.reset_index()).set_index("sample_id").loc[sids_c]
    groups = {col: {v: np.nonzero(df[col].values == v)[0] for v in sorted(df[col].unique())}
              for col in cfg.slices}

    point_c = metric_set(rec_c, groups, df)
    report = {"candidate": _card_summary(card_c), "baseline": _card_summary(card_b) if card_b else None,
              "dataset_version": manifest["version"], "test_frames": len(sids_c),
              "metrics": {"candidate": point_c}, "checks": [], "decision": None}
    checks = report["checks"]
    checks.append({"check": "min_mAP", "value": point_c["mAP"], "threshold": cfg.min_map,
                   "passed": point_c["mAP"] >= cfg.min_map})
    lineage_ok = all(card_c.get(k) for k in ("dataset_version", "weights_sha256", "raw_inputs", "code_version"))
    checks.append({"check": "lineage_complete", "passed": bool(lineage_ok)})

    if rec_b is not None:
        assert sids_b == sids_c, "baseline and candidate evaluated on different frames"
        point_b = metric_set(rec_b, groups, df)
        report["metrics"]["baseline"] = point_b
        rng = np.random.default_rng(cfg.seed)
        n = len(sids_c)
        boots = {k: [] for k in point_c}
        for _ in range(cfg.n_boot):
            s = rng.integers(0, n, n)
            bc = metric_set([rec_c[i] for i in s], _regroup(groups, s), df)
            bb = metric_set([rec_b[i] for i in s], _regroup(groups, s), df)
            for k in boots:
                if k in bc and k in bb and not (np.isnan(bc[k]) or np.isnan(bb[k])):
                    boots[k].append(bc[k] - bb[k])
        lo_q, hi_q = (1 - cfg.ci) / 2, 1 - (1 - cfg.ci) / 2
        deltas = {}
        for k, v in boots.items():
            if not v:
                continue
            d = point_c[k] - point_b[k]
            lo, hi = np.quantile(v, [lo_q, hi_q])
            deltas[k] = {"delta": float(d), "ci_low": float(lo), "ci_high": float(hi)}
            tol = cfg.overall_tolerance if k == "mAP" else cfg.slice_tolerance
            # block only on a regression that is both larger than tolerance and
            # statistically distinguishable from zero
            regression = d < -tol and hi < 0
            checks.append({"check": f"no_regression[{k}]", "delta": round(float(d), 4),
                           "ci": [round(float(lo), 4), round(float(hi), 4)], "tolerance": tol,
                           "passed": not regression})
        report["deltas"] = deltas
    report["decision"] = "promote" if all(c["passed"] for c in checks) else "block"
    return report


def _regroup(groups, s):
    """Slice membership for a bootstrap resample s (positions into the resample)."""
    return {col: {v: np.nonzero(np.isin(s, rows))[0] for v, rows in g.items()}
            for col, g in groups.items()}


def _card_summary(card):
    if not card:
        return None
    return {k: card.get(k) for k in ("mlflow_run_id", "registered_version", "dataset_version",
                                      "weights_sha256", "code_version")} | {"inputs": card["config"]["inputs"],
                                                                               "run_name": card["config"]["run_name"]}


def render_markdown(report: dict) -> str:
    c = report["candidate"]
    lines = [f"# Validation report: {c['run_name']} ({report['decision'].upper()})", "",
             f"- dataset version: `{report['dataset_version']}` ({report['test_frames']} test frames)",
             f"- candidate weights: `{c['weights_sha256'][:16]}` trained on `{c['dataset_version']}`"]
    if report.get("baseline"):
        lines.append(f"- baseline: {report['baseline']['run_name']} `{report['baseline']['weights_sha256'][:16]}`")
    lines += ["", "| check | value / delta | 95% CI | result |", "|---|---|---|---|"]
    for ch in report["checks"]:
        val = ch.get("delta", ch.get("value", ""))
        val = f"{val:+.3f}" if isinstance(val, float) and "delta" in ch else (f"{val:.3f}" if isinstance(val, float) else val)
        ci = f"[{ch['ci'][0]:+.3f}, {ch['ci'][1]:+.3f}]" if "ci" in ch else ""
        lines.append(f"| {ch['check']} | {val} | {ci} | {'pass' if ch['passed'] else '**FAIL**'} |")
    return "\n".join(lines) + "\n"


def promote(card_dir, tracking_uri: str, model_name: str, store=None, alias="production"):
    """Points the registry alias (and a store ref) at the candidate."""
    card = json.loads((Path(card_dir) / "model_card.json").read_text())
    if card.get("registered_version"):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(tracking_uri)
        MlflowClient().set_registered_model_alias(model_name, alias, card["registered_version"])
    if store is not None:
        store.put_ref(f"models/{model_name}/{alias}", {"model_dir": str(card_dir), **card})
    return card


def gate(candidate_dir, baseline_dir, store, dataset: str, version: str, cfg: GateConfig,
         enrichment=None) -> dict:
    manifest = load_manifest(store, dataset, version)
    return compare(candidate_dir, baseline_dir, store, manifest, cfg, enrichment)
