"""Data quality checks run at ingestion, before anything reaches a training set.

Every check is a pure function of the raw log, so results are reproducible
and can be scored against the simulator's hidden ground truth.
"""
from __future__ import annotations

import numpy as np

from ..sim.world import CLASSES
from .raw import RawFlight
from .sync import estimate_lidar_offset, gsd

# Physical size envelopes (metres) for axis-aligned boxes that are not clipped
# by the image border. Boxes outside the envelope are flagged as suspect labels.
SIZE_RULES = {
    "vehicle": {"max_dim": (2.5, 12.5), "min_dim": (1.0, 9.0)},
    "person": {"max_dim": (0.3, 1.4), "min_dim": (0.2, 1.4)},
    "building": {"max_dim": (6.0, 45.0), "min_dim": (4.0, 45.0)},
}
SYNC_MIN_PROMINENCE = 0.045   # alignment-curve peak prominence needed to trust an estimate
SYNC_MIN_OFFSET_S = 0.04      # smaller offsets are left alone


def camera_fault(img: np.ndarray) -> str | None:
    m, s = float(img.mean()), float(img.std())
    if m < 8 and s < 3:
        return "dead"
    if m > 245 and s < 6:
        return "saturated"
    return None


def lidar_gaps(lidar_ts_us: np.ndarray, rate_hz: float) -> list[tuple[float, float]]:
    t = lidar_ts_us / 1e6
    d = np.diff(t)
    idx = np.nonzero(d > 1.5 / rate_hz)[0]
    return [(float(t[i]), float(t[i + 1])) for i in idx]


def check_boxes(boxes: list[dict], alt: float, img_size: int = 96) -> list[str | None]:
    """Per-box issue (None if plausible)."""
    g = gsd(alt)
    issues = []
    for b in boxes:
        x0, x1 = b["x"] - b["w"] / 2, b["x"] + b["w"] / 2
        y0, y1 = b["y"] - b["h"] / 2, b["y"] + b["h"] / 2
        if b["cls"] not in range(len(CLASSES)):
            issues.append("unknown_class")
            continue
        if x0 < -0.5 or y0 < -0.5 or x1 > img_size + 0.5 or y1 > img_size + 0.5 or b["w"] <= 0 or b["h"] <= 0:
            issues.append("out_of_bounds")
            continue
        clipped = x0 <= 0.5 or y0 <= 0.5 or x1 >= img_size - 0.5 or y1 >= img_size - 0.5
        if clipped:
            issues.append(None)
            continue
        dims = sorted([b["w"] * g, b["h"] * g])
        rule = SIZE_RULES[CLASSES[b["cls"]]]
        lo, hi = rule["max_dim"]
        lo2, hi2 = rule["min_dim"]
        if not (lo <= dims[1] <= hi and lo2 <= dims[0] <= hi2):
            issues.append("implausible_size")
        else:
            issues.append(None)
    return issues


def run_flight_qc(flight: RawFlight, estimate_sync: bool = True) -> dict:
    meta = flight.meta
    rate = meta["sensors"]["lidar"]["rate_hz"]
    report = {"flight_id": flight.flight_id, "frames": {}, "issues": []}

    odom_dt = np.diff(flight.odom[:, 0])
    if (odom_dt <= 0).any():
        report["issues"].append("odometry_non_monotonic")

    gaps = lidar_gaps(flight.lidar_ts, rate)
    report["lidar_gaps"] = gaps
    if any(b - a > 1.0 for a, b in gaps):
        report["issues"].append("lidar_gap_over_1s")

    good_frames = []
    for ts in flight.camera_ts:
        ts = int(ts)
        img = flight.image(ts)
        fault = camera_fault(img)
        t = ts / 1e6
        alt = float(np.interp(t, flight.odom[:, 0], flight.odom[:, 3]))
        boxes = flight.labels.get(ts, [])
        box_issues = check_boxes(boxes, alt)
        report["frames"][ts] = {
            "camera_fault": fault,
            "brightness": round(float(img.mean()), 2),
            "contrast": round(float(img.std()), 2),
            "box_issues": box_issues,
            "has_labels": ts in flight.labels,
        }
        if fault is None:
            good_frames.append(ts)

    if estimate_sync and len(flight.lidar_ts) and good_frames:
        est = estimate_lidar_offset(flight, good_frames)
        if est["prominence"] < SYNC_MIN_PROMINENCE:
            status, applied = "unverified", 0.0
        elif abs(est["offset_s"]) < SYNC_MIN_OFFSET_S:
            status, applied = "ok", 0.0
        else:
            status, applied = "corrected", est["offset_s"]
        report["sync"] = {**est, "status": status, "applied_offset_s": applied}
        if status == "corrected":
            report["issues"].append("lidar_clock_offset_corrected")
        elif status == "unverified":
            report["issues"].append("lidar_sync_unverified")
    else:
        report["sync"] = {"status": "skipped", "applied_offset_s": 0.0}

    n = len(report["frames"])
    faults = sum(1 for f in report["frames"].values() if f["camera_fault"])
    suspect = sum(sum(1 for i in f["box_issues"] if i) for f in report["frames"].values())
    total_boxes = sum(len(f["box_issues"]) for f in report["frames"].values())
    report["summary"] = {
        "frames": n, "camera_faults": faults, "boxes": total_boxes,
        "suspect_boxes": suspect, "lidar_sweeps": int(len(flight.lidar_ts)),
        "lidar_gaps": len(gaps),
    }
    return report
