"""Synthetic UxV survey flights: camera, LiDAR, odometry and annotations.

Writes raw sensor logs laid out the way a real flight-log dump looks
(one folder per flight, one file per frame / sweep, timestamps in filenames)
and, separately, a ground-truth file the platform never reads. The truth file
is only used by the benchmarks to score quality checks, label-error detection
and metadata enrichment.

Realistic defects are injected on purpose: LiDAR clock offsets, dropped
sweeps, dead / saturated camera frames and noisy third-party annotations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .world import (
    BUILDING, CLASSES, GROUND_INTENSITY, OBJECT_INTENSITY, ROAD_INTENSITY, TREE,
    World, ground_rgb, is_road, make_world, object_masks,
)

IMG = 96            # camera resolution (square)
FOV_DEG = 50.0      # nadir camera field of view
CAM_HZ = 5.0
LIDAR_HZ = 10.0
ODOM_HZ = 50.0
LIDAR_POINTS = 3000
TOD_BRIGHTNESS = {"day": 1.0, "dusk": 0.5, "night": 0.16}
TOD_NOISE = {"day": 0.008, "dusk": 0.025, "night": 0.05}
TOD_EXPOSURE = {"day": 1 / 800, "dusk": 1 / 120, "night": 1 / 30}


@dataclass
class FlightSpec:
    flight_id: str
    seed: int
    duration_s: float
    tod: str
    fog: float
    altitude: float
    speed: float
    lidar_clock_offset_s: float
    lidar_gap: bool
    noisy_annotations: bool


def gsd(altitude: float) -> float:
    return 2 * altitude * np.tan(np.radians(FOV_DEG / 2)) / IMG


def pixel_grid(x, y, alt, yaw, n=IMG, ss=1):
    """World coordinates of pixel centres (n*ss x n*ss) for a nadir camera."""
    g = gsd(alt) / ss
    m = n * ss
    idx = (np.arange(m) + 0.5) - m / 2
    right = idx[None, :] * g
    fwd = -idx[:, None] * g
    fx, fy = np.cos(yaw), np.sin(yaw)
    rx, ry = np.sin(yaw), -np.cos(yaw)
    X = x + fwd * fx + right * rx
    Y = y + fwd * fy + right * ry
    return X, Y


def world_to_image(px, py, x, y, alt, yaw):
    """Continuous image coordinates (u right, v down; 0..IMG) of world points."""
    g = gsd(alt)
    dx, dy = px - x, py - y
    fwd = dx * np.cos(yaw) + dy * np.sin(yaw)
    right = dx * np.sin(yaw) - dy * np.cos(yaw)
    return right / g + IMG / 2, IMG / 2 - fwd / g


def simulate_trajectory(spec: FlightSpec, world: World, rng):
    n = int(spec.duration_s * ODOM_HZ) + 1
    t = np.arange(n) / ODOM_HZ
    margin = 50.0
    x, y = rng.uniform(margin, world.size - margin, 2)
    yaw = rng.uniform(-np.pi, np.pi)
    yaw_rate = 0.0
    xs, ys, yaws = np.empty(n), np.empty(n), np.empty(n)
    dt = 1 / ODOM_HZ
    for i in range(n):
        xs[i], ys[i], yaws[i] = x, y, yaw
        # steer back towards the centre near the edges, otherwise wander
        to_c = np.arctan2(world.size / 2 - y, world.size / 2 - x)
        edge = min(x, y, world.size - x, world.size - y)
        if edge < margin:
            err = (to_c - yaw + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = np.clip(0.8 * err, -0.35, 0.35)
        else:
            yaw_rate = np.clip(yaw_rate + rng.normal(0, 0.02), -0.15, 0.15)
        yaw += yaw_rate * dt
        x += spec.speed * np.cos(yaw) * dt
        y += spec.speed * np.sin(yaw) * dt
    alt = spec.altitude + 1.5 * np.sin(2 * np.pi * t / 23.0) + rng.normal(0, 0.05, n)
    return t, xs, ys, alt, np.unwrap(yaws)


def render_camera(world: World, x, y, alt, yaw, spec: FlightSpec, rng):
    ss = 2
    X, Y = pixel_grid(x, y, alt, yaw, ss=ss)
    img = ground_rgb(world, X, Y)
    idmap = np.full(X.shape, -1, dtype=np.int64)
    half = alt * np.tan(np.radians(FOV_DEG / 2)) * 1.5
    near = world.objects_near(x, y, half)
    if len(near):
        near = near[np.argsort(world.obj["h"][near])]  # painter's order: tallest on top
        masks = object_masks(world, near, X, Y)
        for k, oid in enumerate(near):
            m = masks[k]
            if not m.any():
                continue
            col = world.obj["rgb"][oid]
            if world.obj["kind"][oid] == BUILDING:  # simple roof shading
                shade = 0.9 + 0.1 * np.sign(np.sin(world.obj["yaw"][oid]))
                col = col * shade
            if world.obj["kind"][oid] == TREE:
                col = col * (0.85 + 0.3 * rng.random())
            img[m] = col
            idmap[m] = oid

    b = TOD_BRIGHTNESS[spec.tod]
    img = img * b
    if spec.fog > 0:
        img = img * (1 - spec.fog) + spec.fog * 0.72 * b
    # downsample (anti-aliasing)
    img = img.reshape(IMG, ss, IMG, ss, 3).mean(axis=(1, 3))
    # motion blur along the direction of travel (image vertical axis)
    blur_px = spec.speed * TOD_EXPOSURE[spec.tod] / gsd(alt)
    k = int(round(blur_px))
    if k >= 2:
        kernel = np.ones(k) / k
        img = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, img)
    noise = TOD_NOISE[spec.tod]
    img = img + rng.normal(0, noise, img.shape) + rng.normal(0, noise / 2, (1, 1, 3))
    img = np.clip(img, 0, 1)
    return (img * 255 + 0.5).astype(np.uint8), idmap, near


def frame_labels(world: World, idmap, near, x, y, alt, yaw):
    """Visible target objects as axis-aligned boxes (centre x, y, w, h in pixels)."""
    g = gsd(alt)
    out = []
    if len(near) == 0:
        return out
    ids, counts = np.unique(idmap[idmap >= 0], return_counts=True)
    vis = dict(zip(ids.tolist(), (counts / 4.0).tolist()))  # 2x2 supersampling
    o = world.obj
    for oid in near:
        cls = int(o["kind"][oid])
        if cls >= len(CLASSES):
            continue
        v = vis.get(int(oid), 0.0)
        w, l = o["w"][oid], o["l"][oid]
        circular = cls == 1
        area_px = (np.pi * (w / 2) ** 2 if circular else w * l) / g**2
        if v < 0.5 or v < 0.3 * area_px:
            continue
        if circular:
            cu, cv = world_to_image(o["cx"][oid], o["cy"][oid], x, y, alt, yaw)
            r = w / 2 / g
            u0, u1, v0, v1 = cu - r, cu + r, cv - r, cv + r
        else:
            c, s = np.cos(o["yaw"][oid]), np.sin(o["yaw"][oid])
            corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, w / 2], [-l / 2, -w / 2]])
            px = o["cx"][oid] + corners[:, 0] * c - corners[:, 1] * s
            py = o["cy"][oid] + corners[:, 0] * s + corners[:, 1] * c
            uu, vv = world_to_image(px, py, x, y, alt, yaw)
            u0, u1, v0, v1 = uu.min(), uu.max(), vv.min(), vv.max()
        u0, u1 = max(u0, 0.0), min(u1, float(IMG))
        v0, v1 = max(v0, 0.0), min(v1, float(IMG))
        if u1 <= u0 or v1 <= v0:
            continue
        out.append({"cls": cls, "x": round((u0 + u1) / 2, 2), "y": round((v0 + v1) / 2, 2),
                    "w": round(u1 - u0, 2), "h": round(v1 - v0, 2), "obj": int(oid)})
    return out


def lidar_sweep(world: World, x, y, alt, yaw, fog, rng):
    """Deskewed sweep in the sensor frame: (N, 4) float32 = forward, left, up, intensity."""
    half = alt * np.tan(np.radians(FOV_DEG / 2))
    R = 1.6 * half
    n = int(LIDAR_POINTS * (1 - 0.45 * fog))
    r = R * np.sqrt(rng.random(n))
    th = rng.uniform(0, 2 * np.pi, n)
    px, py = x + r * np.cos(th), y + r * np.sin(th)
    h = np.zeros(n)
    inten = np.where(is_road(world, px, py), ROAD_INTENSITY, GROUND_INTENSITY)
    near = world.objects_near(x, y, R)
    if len(near):
        masks = object_masks(world, near, px, py)  # (K, n)
        hk = np.where(masks, world.obj["h"][near][:, None], -1.0)
        top = hk.argmax(0)
        hit = hk.max(0) > 0
        h[hit] = hk.max(0)[hit]
        inten[hit] = OBJECT_INTENSITY[world.obj["kind"][near][top[hit]]]
    fwd = (px - x) * np.cos(yaw) + (py - y) * np.sin(yaw)
    left = -((px - x) * np.sin(yaw) - (py - y) * np.cos(yaw))
    up = h - alt + rng.normal(0, 0.03, n)
    pts = np.column_stack([fwd, left, up, inten + rng.normal(0, 0.03, n)])
    if fog > 0:  # backscatter: spurious short-range returns
        m = int(600 * fog)
        rr = rng.uniform(0, R * 0.5, m)
        tt = rng.uniform(0, 2 * np.pi, m)
        fogpts = np.column_stack([rr * np.cos(tt), rr * np.sin(tt),
                                  -rng.uniform(2, 15, m), rng.uniform(0.0, 0.1, m)])
        pts = np.vstack([pts, fogpts])
    return pts.astype(np.float32)


def corrupt_annotations(labels, rng, truth_events, ts):
    """Simulates an annotation vendor: missed boxes, sloppy boxes, wrong classes."""
    out = []
    for lb in labels:
        u = rng.random()
        if u < 0.12:
            truth_events.append({"ts_us": ts, "obj": lb["obj"], "error": "missing"})
            continue
        lb = dict(lb)
        if u < 0.22:
            ang = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(2.5, 5.0) + 0.25 * max(lb["w"], lb["h"])
            lb["x"] = round(float(np.clip(lb["x"] + d * np.cos(ang), 0, IMG)), 2)
            lb["y"] = round(float(np.clip(lb["y"] + d * np.sin(ang), 0, IMG)), 2)
            truth_events.append({"ts_us": ts, "obj": lb["obj"], "error": "shifted"})
        elif u < 0.26:
            lb["cls"] = int((lb["cls"] + rng.integers(1, len(CLASSES))) % len(CLASSES))
            truth_events.append({"ts_us": ts, "obj": lb["obj"], "error": "wrong_class"})
        out.append(lb)
    return out


def generate_flight(spec: FlightSpec, raw_root: Path, truth_root: Path) -> dict:
    rng = np.random.default_rng(spec.seed)
    world = make_world(rng)
    t, xs, ys, alt, yaw = simulate_trajectory(spec, world, rng)
    fdir = raw_root / spec.flight_id
    (fdir / "camera").mkdir(parents=True, exist_ok=True)
    (fdir / "lidar").mkdir(parents=True, exist_ok=True)

    def pose_at(tq):
        return (np.interp(tq, t, xs), np.interp(tq, t, ys),
                np.interp(tq, t, alt), np.interp(tq, t, yaw))

    # odometry (fused GNSS/INS): small noise
    with open(fdir / "odometry.csv", "w") as f:
        f.write("ts_us,x,y,z,yaw\n")
        for i in range(len(t)):
            f.write(f"{int(round(t[i] * 1e6))},{xs[i] + rng.normal(0, 0.05):.3f},"
                    f"{ys[i] + rng.normal(0, 0.05):.3f},{alt[i]:.3f},"
                    f"{yaw[i] + rng.normal(0, 0.002):.5f}\n")

    truth = {"spec": asdict(spec), "camera_faults": [], "dropped_sweeps": [],
             "annotation_errors": [], "frames": []}

    # LiDAR (own clock: recorded timestamp = true time + offset)
    n_sweeps = int(spec.duration_s * LIDAR_HZ)
    lo = min(5.0, 0.2 * spec.duration_s)  # (short test flights: keep the window valid)
    gap_start = rng.uniform(lo, max(lo, spec.duration_s - 8)) if spec.lidar_gap else -1
    for k in range(n_sweeps):
        tt = k / LIDAR_HZ + 0.013
        if spec.lidar_gap and gap_start <= tt < gap_start + 2.0:
            truth["dropped_sweeps"].append(round(tt, 3))
            continue
        if rng.random() < 0.02:
            truth["dropped_sweeps"].append(round(tt, 3))
            continue
        px, py, pa, pyw = pose_at(tt)
        pts = lidar_sweep(world, px, py, pa, pyw, spec.fog, rng)
        ts_rec = int(round((tt + spec.lidar_clock_offset_s) * 1e6))
        pts.tofile(fdir / "lidar" / f"{ts_rec}.bin")

    # Camera + annotations
    labels_out = open(fdir / "labels.jsonl", "w")
    n_frames = int(spec.duration_s * CAM_HZ)
    for k in range(n_frames):
        tt = k / CAM_HZ + 0.05 + rng.normal(0, 0.002)
        ts = int(round(tt * 1e6))
        px, py, pa, pyw = pose_at(tt)
        img, idmap, near = render_camera(world, px, py, pa, pyw, spec, rng)
        labels = frame_labels(world, idmap, near, px, py, pa, pyw)
        u = rng.random()
        if u < 0.006:
            img[:] = rng.integers(0, 4, img.shape)
            truth["camera_faults"].append({"ts_us": ts, "fault": "dead"})
        elif u < 0.010:
            img[:] = 255 - rng.integers(0, 3, img.shape)
            truth["camera_faults"].append({"ts_us": ts, "fault": "saturated"})
        Image.fromarray(img).save(fdir / "camera" / f"{ts}.png", optimize=False, compress_level=1)
        ann = (corrupt_annotations(labels, rng, truth["annotation_errors"], ts)
               if spec.noisy_annotations else labels)
        labels_out.write(json.dumps({"ts_us": ts, "boxes": [
            {k2: b[k2] for k2 in ("cls", "x", "y", "w", "h")} for b in ann]}) + "\n")
        truth["frames"].append({"ts_us": ts, "boxes": labels,
                                "pose": [round(float(v), 3) for v in (px, py, pa, pyw)]})
    labels_out.close()

    meta = {
        "flight_id": spec.flight_id,
        "platform": "uxv-sim",
        "sensors": {
            "camera": {"model": "nadir-rgb", "width": IMG, "height": IMG,
                       "fov_deg": FOV_DEG, "rate_hz": CAM_HZ},
            "lidar": {"model": "spinning-3d", "rate_hz": LIDAR_HZ,
                      "frame": "forward-left-up", "format": "float32[N,4] x,y,z,intensity"},
            "odometry": {"rate_hz": ODOM_HZ, "source": "gnss-ins"},
        },
        "calibration": {"lidar_to_body": "identity", "camera_to_body": "nadir, image-up = forward"},
        "annotation_vendor": "vendor-a" if spec.noisy_annotations else "vendor-b",
    }
    (fdir / "meta.json").write_text(json.dumps(meta, indent=2))
    truth_root.mkdir(parents=True, exist_ok=True)
    (truth_root / f"{spec.flight_id}.json").write_text(json.dumps(truth))
    return meta


def plan_flights(n_flights: int, seed: int, duration_s: float = 32.0) -> list[FlightSpec]:
    rng = np.random.default_rng(seed)
    tods = ["day", "dusk", "night"]
    specs = []
    for i in range(n_flights):
        sync = rng.random() < 0.25
        specs.append(FlightSpec(
            flight_id=f"flight_{i:04d}",
            seed=int(rng.integers(2**31)),
            duration_s=duration_s,
            tod=tods[i % 3],
            fog=float(rng.choice([0.0, 0.0, 0.0, 0.3, 0.55])),
            altitude=round(float(rng.uniform(25, 60)), 1),
            speed=round(float(rng.uniform(6, 14)), 1),
            lidar_clock_offset_s=round(float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.2)), 3) if sync else 0.0,
            lidar_gap=bool(rng.random() < 0.2),
            noisy_annotations=bool(rng.random() < 0.25),
        ))
    return specs
