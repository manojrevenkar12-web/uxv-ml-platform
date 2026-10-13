"""Benchmark training runs (see docs/RESULTS.md). Usage: python scripts/train_runs.py WORKDIR [run ...]"""
import json
import sys
from pathlib import Path

from uxvml.data.dataset import IngestParams, build_dataset
from uxvml.data.store import open_store
from uxvml.train.trainer import TrainConfig, train


def main():
    work = Path(sys.argv[1])
    store_uri = str(work / "store")
    tracking = f"sqlite:///{work / 'mlflow.db'}"
    store = open_store(store_uri)

    RUNS = {
        "fusion": TrainConfig(run_name="fusion", inputs="rgb+lidar"),
        "camera-only": TrainConfig(run_name="camera-only", inputs="rgb"),
        "fusion-repro": TrainConfig(run_name="fusion-repro", inputs="rgb+lidar"),
        "fusion-nosync": TrainConfig(run_name="fusion-nosync", dataset="uxv-nosync", inputs="rgb+lidar"),
        "fusion-trusted-labels": TrainConfig(run_name="fusion-trusted-labels", inputs="rgb+lidar",
                                             exclude_vendors=["vendor-a"]),
        # seed-to-seed spread: the training noise floor the frame bootstrap does not see
        "fusion-seed1": TrainConfig(run_name="fusion-seed1", inputs="rgb+lidar", seed=1),
        "fusion-seed2": TrainConfig(run_name="fusion-seed2", inputs="rgb+lidar", seed=2),
    }

    wanted = sys.argv[2:] or list(RUNS)
    if "fusion-nosync" in wanted and not store.has_ref("datasets/uxv-nosync/latest"):
        build_dataset(work / "sim" / "raw", store, "uxv-nosync", IngestParams(sync_correction=False))
    for name in wanted:
        card = train(RUNS[name], store_uri, tracking, str(work / "models" / name))
        print(json.dumps({"run": name, "val_mAP": card["val_mAP"], "weights": card["weights_sha256"][:16]}), flush=True)


if __name__ == "__main__":
    main()
