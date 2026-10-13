"""Cross-sensor time synchronisation and camera/LiDAR fusion rasters.

LiDAR points are motion-compensated with odometry: a sweep is placed in the
world using the vehicle pose at the sweep's timestamp, then projected into the
camera using the pose at the image timestamp. If the LiDAR clock is offset from
the camera/odometry clock the projection lands in the wrong place, so the
offset is estimated from the data itself (target-less camera/LiDAR alignment).
"""
from __future__ import annotations

import numpy as np

from ..sim.flight import FOV_DEG, IMG
from .raw import RawFlight

RASTER = 48               # LiDAR raster resolution (image is 96x96)
MATCH_TOL_S = 0.06        # max |camera - lidar| time difference after correction


def gsd(alt):
    return 2 * alt * np.tan(np.radians(FOV_DEG / 2)) / IMG


def nearest_sweep(lidar_ts_s: np.ndarray, t: float):
    if len(lidar_ts_s) == 0:
        return None, np.inf
    i = int(np.searchsorted(lidar_ts_s, t))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(lidar_ts_s):
            if best is None or abs(lidar_ts_s[j] - t) < abs(lidar_ts_s[best] - t):
                best = j
    return best, abs(lidar_ts_s[best] - t)


def project_sweep(pts: np.ndarray, sweep_pose, cam_pose):
    """Sensor-frame points -> continuous image coords (u, v) and height above ground."""
    sx, sy, sz, syaw = sweep_pose
    cx, cy, cz, cyaw = cam_pose
    fwd, left, up = pts[:, 0], pts[:, 1], pts[:, 2]
    X = sx + fwd * np.cos(syaw) - left * np.sin(syaw)
    Y = sy + fwd * np.sin(syaw) + left * np.cos(syaw)
    h = up + sz
    g = gsd(cz)
    dx, dy = X - cx, Y - cy
    f = dx * np.cos(cyaw) + dy * np.sin(cyaw)
    r = dx * np.sin(cyaw) - dy * np.cos(cyaw)
    return r / g + IMG / 2, IMG / 2 - f / g, h


def rasterize(u, v, h, size=RASTER):
    """(2, size, size): max height above ground (m) and log1p(point count)."""
    s = size / IMG
    iu = np.floor(u * s).astype(np.int64)
    iv = np.floor(v * s).astype(np.int64)
    ok = (iu >= 0) & (iu < size) & (iv >= 0) & (iv < size) & (h > -1.0)
    iu, iv, hh = iu[ok], iv[ok], np.clip(h[ok], 0, 30)
    height = np.zeros((size, size), np.float32)
    count = np.zeros((size, size), np.float32)
    np.maximum.at(height, (iv, iu), hh)
    np.add.at(count, (iv, iu), 1)
    return np.stack([height, np.log1p(count)])


def fusion_raster(flight: RawFlight, cam_ts_us: int, offset_s: float, lidar_ts_s=None):
    """LiDAR raster aligned to a camera frame; returns (raster, sweep_ts_us or None, dt)."""
    if lidar_ts_s is None:
        lidar_ts_s = flight.lidar_ts / 1e6 - offset_s
    t_cam = cam_ts_us / 1e6
    j, dt = nearest_sweep(lidar_ts_s, t_cam)
    if j is None or dt > MATCH_TOL_S:
        return np.zeros((2, RASTER, RASTER), np.float32), None, float(dt)
    pts = flight.sweep(int(flight.lidar_ts[j]))
    u, v, h = project_sweep(pts, flight.pose(lidar_ts_s[j]), flight.pose(t_cam))
    return rasterize(u, v, h), int(flight.lidar_ts[j]), float(dt)


def _grad(a):
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.hypot(gx, gy)


def _image_edges(img: np.ndarray) -> np.ndarray:
    g = img.astype(np.float32).mean(-1)
    g = g.reshape(RASTER, IMG // RASTER, RASTER, IMG // RASTER).mean((1, 3))
    g = (g - g.mean()) / (g.std() + 1e-6)
    return _grad(g)


def estimate_lidar_offset(flight: RawFlight, frame_ts: list[int],
                          search_s: float = 0.25, step_s: float = 0.01,
                          max_frames: int = 30) -> dict:
    """Finds the LiDAR clock offset that best aligns height edges with image edges.

    Uses the highest-contrast frames (dark or foggy frames carry little signal).
    Returns the offset, the alignment score curve and a confidence margin.
    """
    contrasts = []
    for ts in frame_ts:
        img = flight.image(ts)
        contrasts.append((float(img.astype(np.float32).std()), ts))
    contrasts.sort(reverse=True)
    chosen = [ts for _, ts in contrasts[:max_frames]]
    edges = {ts: _image_edges(flight.image(ts)) for ts in chosen}
    sweeps = {}
    offsets = np.round(np.arange(-search_s, search_s + 1e-9, step_s), 4)
    lid_s = flight.lidar_ts / 1e6
    scores = []
    for off in offsets:
        num, n = 0.0, 0
        corrected = lid_s - off
        for ts in chosen:
            j, dt = nearest_sweep(corrected, ts / 1e6)
            if j is None or dt > MATCH_TOL_S:
                continue
            key = int(flight.lidar_ts[j])
            if key not in sweeps:
                sweeps[key] = flight.sweep(key)
            u, v, h = project_sweep(sweeps[key], flight.pose(corrected[j]), flight.pose(ts / 1e6))
            hg = _grad(rasterize(u, v, h)[0])
            ie = edges[ts]
            denom = np.linalg.norm(hg) * np.linalg.norm(ie)
            if denom > 0:
                num += float((hg * ie).sum() / denom)
                n += 1
        scores.append(num / n if n else 0.0)
    scores = np.array(scores)
    best = int(scores.argmax())
    zero = int(np.argmin(np.abs(offsets)))
    return {
        "offset_s": float(offsets[best]),
        "score_at_offset": float(scores[best]),
        "score_at_zero": float(scores[zero]),
        "gain": float(scores[best] - scores[zero]),
        # peak prominence: how much the best offset stands out from the curve;
        # low-contrast (night / fog) flights give flat curves we must not trust
        "prominence": float(scores[best] - np.median(scores)),
        "frames_used": len(chosen),
    }
