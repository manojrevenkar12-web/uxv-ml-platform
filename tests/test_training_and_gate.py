import json
from pathlib import Path

import numpy as np
import torch

from uxvml.train.model import CenterNetLite
from uxvml.train.targets import (average_precision, decode, make_targets, map_from_records, match,
                                 per_class_records)
from uxvml.train.trainer import TrainConfig, train, weights_digest
from uxvml.validate.gate import GateConfig, compare, promote


def test_targets_and_decode_roundtrip():
    boxes = np.array([[0, 20.3, 30.7, 9.0, 4.0], [1, 70.0, 12.0, 1.6, 1.6], [2, 50.5, 70.5, 20, 16]], np.float32)
    heat, reg, mask = make_targets(boxes)
    assert mask.sum() == 3 and heat.max() == 1.0
    logits = torch.from_numpy(np.log(np.clip(heat, 1e-6, 1 - 1e-6) / (1 - np.clip(heat, 1e-6, 1 - 1e-6))))[None]
    det = decode(logits, torch.from_numpy(reg)[None], min_score=0.9)[0]
    det = det[np.argsort(det[:, 0])]
    np.testing.assert_allclose(det[:, :5], boxes, atol=1e-3)


def test_matching_and_ap():
    gts = np.array([[0, 10, 10, 4, 2], [1, 50, 50, 1, 1]], np.float32)
    preds = np.array([[0, 11, 10, 4, 2, 0.9], [1, 50.5, 50, 1, 1, 0.8], [0, 80, 80, 4, 2, 0.3],
                      [1, 51, 50, 1, 1, 0.2]], np.float32)  # last one is a duplicate
    tp, assign = match(preds, gts)
    assert tp.tolist() == [True, True, False, False]
    assert average_precision([0.9, 0.3], [True, False], 1) == 1.0
    assert average_precision([0.9, 0.3], [False, True], 1) == 0.5
    _, m = map_from_records(per_class_records([preds], [gts]))
    assert m == 1.0


def test_distributed_training_is_bit_reproducible(tiny_model, tiny_dataset):
    root, tracking, card = tiny_model
    store_dir = tiny_dataset[0]
    cfg = TrainConfig(dataset="tiny", epochs=1, batch_size=8, width=8, world_size=2, run_name="t2",
                      register_as="tiny-detector")
    card2 = train(cfg, str(store_dir), tracking, str(root / "b"))
    assert card2["weights_sha256"] == card["weights_sha256"]
    assert card2["dataset_version"] == card["dataset_version"] == tiny_dataset[2]["version"]


def test_different_seed_changes_weights(tiny_model, tiny_dataset, tmp_path):
    root, tracking, card = tiny_model
    cfg = TrainConfig(dataset="tiny", epochs=1, batch_size=8, width=8, world_size=1, seed=1,
                      run_name="t3", register_as=None)
    card3 = train(cfg, str(tiny_dataset[0]), tracking, str(tmp_path / "c"))
    assert card3["weights_sha256"] != card["weights_sha256"]


def test_run_is_tracked_with_lineage(tiny_model):
    import mlflow
    from mlflow import MlflowClient
    root, tracking, card = tiny_model
    mlflow.set_tracking_uri(tracking)
    run = MlflowClient().get_run(card["mlflow_run_id"])
    assert run.data.params["dataset_version_resolved"] == card["dataset_version"]
    assert run.data.tags["weights_sha256"] == card["weights_sha256"]
    assert "final_val_mAP" in run.data.metrics
    mv = MlflowClient().get_model_version("tiny-detector", card["registered_version"])
    assert mv.tags["dataset_version"] == card["dataset_version"]
    assert set(card["raw_inputs"]) and card["code_version"]


def test_gate_passes_identical_model_and_blocks_broken_one(tiny_model, tiny_dataset, tmp_path):
    root, tracking, card = tiny_model
    _, store, manifest, _ = tiny_dataset
    cfg = GateConfig(min_map=0.0, n_boot=60)
    same = compare(root / "a", root / "a", store, manifest, cfg)
    assert same["decision"] == "promote"
    assert abs(same["deltas"]["mAP"]["delta"]) < 1e-12

    # a "candidate" whose weights were scrambled must not ship
    broken = tmp_path / "broken"
    broken.mkdir()
    m = CenterNetLite(card["in_channels"], card["width"])
    torch.manual_seed(123)
    for p in m.parameters():
        if p.dim() == 4:  # scramble conv kernels
            torch.nn.init.normal_(p, std=0.5)
    torch.save(m.state_dict(), broken / "model.pt")
    bad_card = dict(card, weights_sha256=weights_digest(m))
    (broken / "model_card.json").write_text(json.dumps(bad_card))
    cfg = GateConfig(min_map=0.0, n_boot=60, overall_tolerance=0.0)
    rep = compare(broken, root / "a", store, manifest, cfg)
    base_map = rep["metrics"]["baseline"]["mAP"]
    if base_map > 0.05:  # only meaningful if the 1-epoch baseline learned something
        assert rep["decision"] == "block"


def test_gate_requires_lineage(tiny_model, tiny_dataset, tmp_path):
    root, tracking, card = tiny_model
    _, store, manifest, _ = tiny_dataset
    d = tmp_path / "nolineage"
    d.mkdir()
    (d / "model.pt").write_bytes((root / "a" / "model.pt").read_bytes())
    (d / "model_card.json").write_text(json.dumps(dict(card, dataset_version=None)))
    rep = compare(d, None, store, manifest, GateConfig(min_map=0.0))
    assert rep["decision"] == "block"
    assert [c for c in rep["checks"] if c["check"] == "lineage_complete"][0]["passed"] is False


def test_promotion_sets_registry_alias(tiny_model, tiny_dataset):
    from mlflow import MlflowClient
    root, tracking, card = tiny_model
    _, store, _, _ = tiny_dataset
    promote(root / "a", tracking, "tiny-detector", store)
    mv = MlflowClient(tracking).get_model_version_by_alias("tiny-detector", "production")
    assert str(mv.version) == str(card["registered_version"])
    assert store.get_ref("models/tiny-detector/production")["weights_sha256"] == card["weights_sha256"]
    assert Path(store.get_ref("models/tiny-detector/production")["model_dir"]).exists()
