import json
from dataclasses import replace

import numpy as np

from uxvml.data.quality import camera_fault, check_boxes, lidar_gaps, run_flight_qc
from uxvml.data.raw import hash_tree, load_flight
from uxvml.data.sync import estimate_lidar_offset, project_sweep, rasterize
from uxvml.sim.flight import IMG, generate_flight, gsd, plan_flights


def test_simulation_is_deterministic(tmp_path):
    spec = plan_flights(1, 3, duration_s=2.0)[0]
    generate_flight(spec, tmp_path / "a" / "raw", tmp_path / "a" / "truth")
    generate_flight(spec, tmp_path / "b" / "raw", tmp_path / "b" / "truth")
    assert hash_tree(tmp_path / "a" / "raw") == hash_tree(tmp_path / "b" / "raw")


def test_raw_log_layout(tiny_sim):
    sim, specs = tiny_sim
    f = load_flight(sim / "raw" / specs[0].flight_id)
    assert len(f.camera_ts) > 0 and np.all(np.diff(f.camera_ts) > 0)
    assert f.image(int(f.camera_ts[0])).shape == (IMG, IMG, 3)
    pts = f.sweep(int(f.lidar_ts[0]))
    assert pts.shape[1] == 4 and pts.dtype == np.float32
    for boxes in f.labels.values():
        for b in boxes:
            assert 0 <= b["x"] <= IMG and 0 <= b["y"] <= IMG and b["cls"] in (0, 1, 2)


def test_projection_puts_tall_objects_where_the_camera_sees_them(tiny_sim):
    """With correct timestamps, LiDAR heights land on labelled buildings."""
    sim, specs = tiny_sim
    hits = []
    for spec in specs:
        if spec.lidar_clock_offset_s:
            continue
        f = load_flight(sim / "raw" / spec.flight_id)
        lid = f.lidar_ts / 1e6
        for ts, boxes in list(f.labels.items())[::3]:
            j = int(np.argmin(np.abs(lid - ts / 1e6)))
            u, v, h = project_sweep(f.sweep(int(f.lidar_ts[j])), f.pose(lid[j]), f.pose(ts / 1e6))
            ras = rasterize(u, v, h)[0]
            for b in boxes:
                if b["cls"] == 2 and b["w"] > 10 and b["h"] > 10:
                    x0, x1 = int((b["x"] - b["w"] / 4) / 2), int((b["x"] + b["w"] / 4) / 2)
                    y0, y1 = int((b["y"] - b["h"] / 4) / 2), int((b["y"] + b["h"] / 4) / 2)
                    hits.append(ras[y0:y1 + 1, x0:x1 + 1].max() > 3.0)
    assert len(hits) > 5 and np.mean(hits) > 0.8


def test_offset_estimation_recovers_injected_clock_offset(tmp_path):
    spec = replace(plan_flights(6, 11, duration_s=12.0)[1], fog=0.0,
                   lidar_clock_offset_s=0.15, lidar_gap=False)
    generate_flight(spec, tmp_path / "raw", tmp_path / "truth")
    f = load_flight(tmp_path / "raw" / spec.flight_id)
    est = estimate_lidar_offset(f, [int(t) for t in f.camera_ts])
    assert abs(est["offset_s"] - 0.15) <= 0.03
    assert est["prominence"] > 0.045


def test_camera_fault_detection():
    rng = np.random.default_rng(0)
    assert camera_fault(rng.integers(0, 4, (96, 96, 3)).astype(np.uint8)) == "dead"
    assert camera_fault((255 - rng.integers(0, 3, (96, 96, 3))).astype(np.uint8)) == "saturated"
    night = np.clip(rng.normal(15, 12, (96, 96, 3)), 0, 255).astype(np.uint8)
    assert camera_fault(night) is None  # dark but valid


def test_box_size_rules_catch_wrong_classes():
    alt = 40.0
    px = 1 / gsd(alt)
    car = {"cls": 0, "x": 48, "y": 48, "w": 4.6 * px, "h": 2.0 * px}
    person = {"cls": 1, "x": 30, "y": 30, "w": 0.7 * px, "h": 0.7 * px}
    car_as_person = dict(car, cls=1)
    person_as_building = dict(person, cls=2)
    edge_box = {"cls": 1, "x": 1.0, "y": 50, "w": 2.0, "h": 6 * px}   # clipped: not judged
    assert check_boxes([car, person, car_as_person, person_as_building, edge_box], alt) == \
        [None, None, "implausible_size", "implausible_size", None]


def test_lidar_gaps():
    ts = (np.r_[np.arange(0, 1, 0.1), np.arange(2.5, 3, 0.1)] * 1e6).astype(np.int64)
    gaps = lidar_gaps(ts, 10.0)
    assert len(gaps) == 1 and abs(gaps[0][1] - gaps[0][0] - 1.6) < 1e-6


def test_flight_qc_matches_injected_faults(tiny_sim):
    sim, specs = tiny_sim
    for spec in specs:
        f = load_flight(sim / "raw" / spec.flight_id)
        qc = run_flight_qc(f, estimate_sync=False)
        truth = json.loads((sim / "truth" / f"{spec.flight_id}.json").read_text())
        injected = {c["ts_us"] for c in truth["camera_faults"]}
        flagged = {ts for ts, fr in qc["frames"].items() if fr["camera_fault"]}
        assert flagged == injected
