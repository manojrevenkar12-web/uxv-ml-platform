import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def tiny_sim(tmp_path_factory):
    """Twelve short flights (four per lighting condition), generated once per test session."""
    from uxvml.sim import simulate

    out = tmp_path_factory.mktemp("sim")
    specs = simulate(out, n_flights=12, seed=5, duration_s=4.0, workers=2)
    return out, specs


@pytest.fixture(scope="session")
def tiny_dataset(tiny_sim, tmp_path_factory):
    from uxvml.data.dataset import IngestParams, build_dataset
    from uxvml.data.store import open_store

    sim, _ = tiny_sim
    store_dir = tmp_path_factory.mktemp("store")
    store = open_store(store_dir)
    params = IngestParams(shard_size=16, n_val_per_tod=1, n_test_per_tod=1, trusted_vendors=("vendor-a", "vendor-b"))
    manifest = build_dataset(sim / "raw", store, "tiny", params, workers=2, log=lambda *a: None)
    return store_dir, store, manifest, params


@pytest.fixture(scope="session")
def tiny_model(tiny_dataset, tmp_path_factory):
    from uxvml.train.trainer import TrainConfig, train

    store_dir, store, manifest, _ = tiny_dataset
    root = tmp_path_factory.mktemp("models")
    tracking = f"sqlite:///{root / 'mlflow.db'}"
    cfg = TrainConfig(dataset="tiny", epochs=1, batch_size=8, width=8, world_size=2, run_name="t1",
                      register_as="tiny-detector")
    card = train(cfg, str(store_dir), tracking, str(root / "a"))
    return root, tracking, card
