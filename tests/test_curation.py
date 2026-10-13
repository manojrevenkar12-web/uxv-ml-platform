import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from uxvml.curate import autolabel as al
from uxvml.curate.enrich import image_features
from uxvml.curate.search import FrameQuery, LLMQueryParser, parse_query, run_query, similar


@pytest.mark.parametrize("text,expected", [
    ("night frames with at least 3 vehicles", {"lighting": ["night"], "min_counts": {"vehicle": 3}}),
    ("foggy dusk images with people", {"lighting": ["dusk"], "fog": ["light", "heavy"], "min_counts": {"person": 1}}),
    ("heavy fog, more than 2 cars, below 35 m", {"fog": ["heavy"], "min_counts": {"vehicle": 3}, "altitude_max_m": 35.0}),
    ("daytime clear frames without buildings in the test split", {"lighting": ["day"], "fog": ["none"],
                                                                  "max_counts": {"building": 0}, "split": ["test"]}),
    ("crowded scenes at high altitude", {"min_counts": {"person": 5}, "altitude_min_m": 48.0}),
    ("frames with missing lidar and suspect labels", {"lidar": "missing", "suspect_labels": True}),
    ("at most two vehicles and five+ people", {"max_counts": {"vehicle": 2}, "min_counts": {"person": 5}}),
    ("fewer than 4 people above 50 meters", {"max_counts": {"person": 3}, "altitude_min_m": 50.0}),
])
def test_parse_query(text, expected):
    q = parse_query(text).model_dump(exclude_defaults=True)
    assert q == expected


def _table():
    return pd.DataFrame({
        "sample_id": [f"f{i // 3}/{i}" for i in range(9)],
        "lighting": ["night"] * 3 + ["day"] * 3 + ["dusk"] * 3,
        "fog": ["none", "light", "heavy"] * 3,
        "pred_vehicle": [0, 3, 5, 1, 4, 0, 2, 2, 9], "pred_person": [0] * 8 + [7],
        "pred_building": [1] * 9, "altitude_m": [30, 40, 50] * 3,
        "lidar_ok": [True] * 8 + [False], "suspect_boxes": [0] * 9, "split": ["train"] * 9,
    })


def test_run_query_filters():
    t = _table()
    got = run_query(t, parse_query("night frames with at least 3 vehicles"))
    assert got.sample_id.tolist() == ["f0/1", "f0/2"]
    got = run_query(t, parse_query("crowded frames with missing lidar"))
    assert got.sample_id.tolist() == ["f2/8"]
    assert len(run_query(t, FrameQuery(limit=2))) == 2


def test_llm_parser_validates_model_output():
    def client(text):
        return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(text=text)])))
    ok = LLMQueryParser(client('Sure: {"lighting": ["night"], "min_counts": {"vehicle": 2}}'))("x")
    assert ok.lighting == ["night"] and ok.min_counts == {"vehicle": 2}
    with pytest.raises(ValidationError):
        LLMQueryParser(client('{"lighting": ["midnight"]}'))("x")       # not an allowed value
    with pytest.raises(ValidationError):
        LLMQueryParser(client('{"min_counts": {"tank": 1}}'))("x")      # unknown class


def test_similar_excludes_query_flight():
    emb = np.array([[1, 0], [0.99, 0.1], [0.98, 0.0], [0, 1]], np.float32)
    order = ["a/1", "a/2", "b/1", "c/1"]
    assert similar(emb, order, "a/1", k=2) == [("b/1", pytest.approx(1.0, abs=1e-3)), ("c/1", pytest.approx(0.0, abs=1e-3))]


def test_image_features_separate_haze_and_darkness():
    rng = np.random.default_rng(0)
    scene = rng.uniform(0.05, 0.6, (96, 96, 3))
    clear = (scene * 255).astype(np.uint8)
    foggy = ((scene * 0.45 + 0.55 * 0.72) * 255).astype(np.uint8)
    dark = (scene * 0.16 * 255).astype(np.uint8)
    fc, ff, fd = image_features(clear), image_features(foggy), image_features(dark)
    assert ff["dark_ratio"] > fc["dark_ratio"] and ff["contrast"] < fc["contrast"]
    assert fd["lum"] < fc["lum"] / 3


def test_autolabel_routes_uncertain_frames(monkeypatch):
    preds = [np.array([[0, 10, 10, 4, 2, 0.95]], np.float32),          # confident -> auto
             np.array([[0, 10, 10, 4, 2, 0.95], [1, 50, 50, 1, 1, 0.3]], np.float32),  # borderline person
             np.zeros((0, 6), np.float32)]                             # empty -> auto (no objects)
    monkeypatch.setattr(al, "predict", lambda *a, **k: (["s1", "s2", "s3"], preds, [None] * 3))
    labels, review = al.autolabel(None, None, None, "rgb", {"s1", "s2", "s3"},
                                  {"vehicle": 0.8, "person": 0.6, "building": 0.8})
    assert review == ["s2"] and set(labels) == {"s1", "s3"}
    assert labels["s1"].shape == (1, 5)


def test_label_audit_flags(monkeypatch):
    gts = [np.array([[0, 10, 10, 4, 2], [1, 40, 40, 1, 1], [0, 80, 80, 4, 2]], np.float32)]
    preds = [np.array([[0, 10.5, 10, 4, 2, 0.9],      # supports GT 0
                       [0, 40, 40, 4, 2, 0.8],        # GT 1 is labelled person, model sees vehicle
                       [2, 20, 70, 15, 15, 0.9]],     # building with no label
                      np.float32)]                    # GT 2 has no support at all
    monkeypatch.setattr(al, "predict", lambda *a, **k: (["s"], preds, gts))
    flags = al.audit_labels(None, None, None, "rgb", {"s"})
    kinds = sorted((f["kind"], f["box"]) for f in flags)
    assert kinds == sorted([("class_mismatch", 1), ("unsupported_box", 2), ("missing_label", None)])


def test_threshold_calibration_hits_target_precision(monkeypatch):
    rng = np.random.default_rng(0)
    preds, gts = [], []
    for _ in range(200):
        g = np.array([[0, 20, 20, 4, 2]], np.float32)
        good = [0, 20, 20, 4, 2, rng.uniform(0.5, 1.0)]
        bad = [0, 70, 70, 4, 2, rng.uniform(0.0, 0.7)]
        preds.append(np.array([good, bad], np.float32))
        gts.append(g)
    monkeypatch.setattr(al, "predict", lambda *a, **k: (None, preds, gts))
    th = al.calibrate_thresholds(None, None, None, "rgb", target_precision=0.95)
    s = np.concatenate([p[:, 5] for p in preds])
    tp = np.array([True, False] * 200)
    assert tp[s >= th["vehicle"]].mean() >= 0.95
    assert th["person"] > 1.0  # no support -> never auto-accept
