"""Auto-labelling experiment (see docs/RESULTS.md).

1. Train M0 on a third of the trusted-vendor training flights ("labelled").
2. Calibrate per-class thresholds on val for 95% precision; auto-label the
   remaining flights ("pool"); frames with borderline detections go to review
   (simulated here by using their human labels).
3. Train M1 on labelled + pool using those labels and compare it with M0 and
   with a model trained on human labels for all of the same flights.

Usage: python scripts/autolabel_experiment.py WORKDIR
"""
import json
import sys
from pathlib import Path

import numpy as np

from uxvml.curate.autolabel import autolabel, calibrate_thresholds
from uxvml.data.dataset import load_index, load_manifest
from uxvml.data.store import open_store
from uxvml.train.targets import match
from uxvml.train.trainer import TrainConfig, load_model, predict, train


def main():
    work = Path(sys.argv[1])
    store_uri = str(work / "store")
    tracking = f"sqlite:///{work / 'mlflow.db'}"
    store = open_store(store_uri)
    manifest = load_manifest(store, "uxv")
    idx = load_index(store, manifest)
    tr = idx[(idx.split == "train") & (idx.vendor == "vendor-b")]
    flights = sorted(tr.flight_id.unique())
    labelled = flights[::3]
    pool = [f for f in flights if f not in labelled]

    m0_dir = work / "models" / "autolabel-m0"
    if not (m0_dir / "model_card.json").exists():
        train(TrainConfig(run_name="autolabel-m0", train_flights=labelled, register_as=None),
              store_uri, tracking, str(m0_dir))
    m0, card0 = load_model(m0_dir)
    th = calibrate_thresholds(m0, store, manifest, "rgb+lidar", target_precision=0.95)
    pool_ids = set(tr[tr.flight_id.isin(pool)].sample_id)
    auto, review = autolabel(m0, store, manifest, "rgb+lidar", pool_ids, th)

    # quality of auto-accepted labels against the human labels of the same frames
    sids, _, gts = predict(m0, store, manifest, "train", "rgb+lidar", include=set(auto))
    gt = dict(zip(sids, gts))
    tp = n_pred = n_gt = 0
    for sid, boxes in auto.items():
        p = np.concatenate([boxes, np.ones((len(boxes), 1), np.float32)], 1)
        t, _ = match(p, gt[sid])
        tp += int(t.sum())
        n_pred += len(boxes)
        n_gt += len(gt[sid])
    override = {sid: b.tolist() for sid, b in auto.items()}
    store.put_ref("labels/autolabel-m0", {"thresholds": th, "labels": override, "review": sorted(review)})

    m1_dir = work / "models" / "autolabel-m1"
    if not (m1_dir / "model_card.json").exists():
        train(TrainConfig(run_name="autolabel-m1", train_flights=flights, label_override="labels/autolabel-m0",
                          register_as=None), store_uri, tracking, str(m1_dir))

    result = {
        "labelled_flights": labelled, "pool_flights": pool,
        "labelled_frames": int(tr.flight_id.isin(labelled).sum()), "pool_frames": len(pool_ids),
        "thresholds": th, "auto_frames": len(auto), "review_frames": len(review),
        "auto_box_precision": tp / max(n_pred, 1), "auto_box_recall": tp / max(n_gt, 1),
    }
    (work / "results").mkdir(exist_ok=True)
    (work / "results" / "autolabel.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
