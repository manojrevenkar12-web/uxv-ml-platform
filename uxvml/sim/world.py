"""Procedural world model for synthetic UxV survey flights.

A flight area is a flat ground plane with roads, agricultural fields and
objects (vehicles, people, buildings, trees). Everything is generated from a
seed so a flight can be reproduced exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ("vehicle", "person", "building")
VEHICLE, PERSON, BUILDING, TREE = 0, 1, 2, 3
CIRCULAR = (PERSON, TREE)

# LiDAR return intensity per material (ground, vehicle, person, building, tree)
GROUND_INTENSITY = 0.35
ROAD_INTENSITY = 0.15
OBJECT_INTENSITY = np.array([0.80, 0.45, 0.55, 0.25])

VEHICLE_COLORS = np.array(
    [[0.92, 0.92, 0.92], [0.08, 0.08, 0.09], [0.70, 0.10, 0.10],
     [0.12, 0.22, 0.60], [0.62, 0.64, 0.66], [0.85, 0.75, 0.15]]
)
PERSON_COLORS = np.array(
    [[0.85, 0.25, 0.20], [0.20, 0.30, 0.75], [0.95, 0.90, 0.20],
     [0.15, 0.15, 0.15], [0.90, 0.90, 0.90], [0.30, 0.65, 0.30]]
)
ROOF_COLORS = np.array(
    [[0.60, 0.22, 0.15], [0.40, 0.40, 0.42], [0.55, 0.50, 0.45], [0.25, 0.27, 0.30]]
)


@dataclass
class World:
    size: float
    roads: np.ndarray          # (R, 3): x0, y0, angle  (infinite lines through area)
    road_width: float
    fields: np.ndarray         # (F, 6): cx, cy, w, l, yaw, stripe_period
    ground_waves: np.ndarray   # (4, 4): kx, ky, phase, amp
    obj: dict                  # column arrays, see make_world

    def objects_near(self, x: float, y: float, radius: float) -> np.ndarray:
        reach = radius + 0.5 * np.hypot(self.obj["w"], self.obj["l"])
        d = np.hypot(self.obj["cx"] - x, self.obj["cy"] - y)
        return np.nonzero(d <= reach)[0]


def _dist_to_roads(x, y, roads):
    if len(roads) == 0:
        return np.full(np.shape(x), np.inf)
    out = np.full(np.shape(x), np.inf)
    for x0, y0, a in roads:
        nx, ny = -np.sin(a), np.cos(a)
        out = np.minimum(out, np.abs((x - x0) * nx + (y - y0) * ny))
    return out


def make_world(rng: np.random.Generator, size: float = 300.0, density: float = 1.0) -> World:
    road_width = 7.0
    n_roads = rng.integers(2, 5)
    roads = np.column_stack([
        rng.uniform(0.2 * size, 0.8 * size, n_roads),
        rng.uniform(0.2 * size, 0.8 * size, n_roads),
        rng.uniform(0, np.pi, n_roads),
    ])
    n_fields = rng.integers(3, 8)
    fields = np.column_stack([
        rng.uniform(0, size, n_fields), rng.uniform(0, size, n_fields),
        rng.uniform(40, 110, n_fields), rng.uniform(40, 110, n_fields),
        rng.uniform(0, np.pi, n_fields), rng.uniform(2.5, 6.0, n_fields),
    ])
    waves = np.column_stack([
        rng.uniform(0.01, 0.08, 4) * rng.choice([-1, 1], 4),
        rng.uniform(0.01, 0.08, 4) * rng.choice([-1, 1], 4),
        rng.uniform(0, 2 * np.pi, 4), rng.uniform(0.02, 0.06, 4),
    ])

    rows = []  # kind, cx, cy, w, l, yaw, h, r, g, b

    # Vehicles: mostly on roads, aligned with the road.
    n_veh = int(rng.integers(45, 80) * density)
    for _ in range(n_veh):
        if rng.random() < 0.85 and len(roads):
            x0, y0, a = roads[rng.integers(len(roads))]
            t = rng.uniform(-size, size)
            lane = rng.choice([-1.8, 1.8])
            cx = x0 + t * np.cos(a) - lane * np.sin(a)
            cy = y0 + t * np.sin(a) + lane * np.cos(a)
            yaw = a + rng.choice([0, np.pi]) + rng.normal(0, 0.05)
        else:  # parked somewhere
            cx, cy, yaw = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(0, np.pi)
        if not (0 <= cx <= size and 0 <= cy <= size):
            continue
        w, l = rng.uniform(1.7, 2.1), rng.uniform(4.0, 5.2)
        if rng.random() < 0.12:  # vans/trucks
            w, l = rng.uniform(2.2, 2.6), rng.uniform(6.0, 9.0)
        rows.append((VEHICLE, cx, cy, w, l, yaw, rng.uniform(1.4, 3.2),
                     *VEHICLE_COLORS[rng.integers(len(VEHICLE_COLORS))]))

    # People: along road edges and scattered.
    n_per = int(rng.integers(40, 80) * density)
    for _ in range(n_per):
        if rng.random() < 0.6 and len(roads):
            x0, y0, a = roads[rng.integers(len(roads))]
            t = rng.uniform(-size, size)
            off = rng.choice([-1, 1]) * rng.uniform(4.5, 7.0)
            cx = x0 + t * np.cos(a) - off * np.sin(a)
            cy = y0 + t * np.sin(a) + off * np.cos(a)
        else:
            cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        if not (0 <= cx <= size and 0 <= cy <= size):
            continue
        d = rng.uniform(0.6, 0.85)
        rows.append((PERSON, cx, cy, d, d, 0.0, rng.uniform(1.6, 1.9),
                     *PERSON_COLORS[rng.integers(len(PERSON_COLORS))]))

    # Buildings: away from roads.
    n_bld = int(rng.integers(18, 35) * density)
    tries = 0
    placed = 0
    while placed < n_bld and tries < 500:
        tries += 1
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        w, l = rng.uniform(8, 22), rng.uniform(8, 28)
        if _dist_to_roads(np.array(cx), np.array(cy), roads) < road_width / 2 + max(w, l) / 2 + 2:
            continue
        rows.append((BUILDING, cx, cy, w, l, rng.uniform(0, np.pi), rng.uniform(4, 16),
                     *ROOF_COLORS[rng.integers(len(ROOF_COLORS))]))
        placed += 1

    # Trees: anywhere off-road (distractors; not a target class).
    n_tree = int(rng.integers(60, 140) * density)
    for _ in range(n_tree):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        if _dist_to_roads(np.array(cx), np.array(cy), roads) < road_width / 2 + 3:
            continue
        d = rng.uniform(3, 8)
        g = rng.uniform(0.22, 0.38)
        rows.append((TREE, cx, cy, d, d, 0.0, rng.uniform(3, 11), 0.10, g, 0.08))

    arr = np.array(rows, dtype=np.float64)
    obj = {
        "kind": arr[:, 0].astype(np.int64),
        "cx": arr[:, 1], "cy": arr[:, 2], "w": arr[:, 3], "l": arr[:, 4],
        "yaw": arr[:, 5], "h": arr[:, 6], "rgb": arr[:, 7:10],
    }
    obj["id"] = np.arange(len(arr))
    return World(size, roads, road_width, fields, waves, obj)


def ground_rgb(world: World, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Albedo of the ground (no objects) at world coordinates X, Y."""
    base = np.empty(X.shape + (3,))
    base[...] = (0.30, 0.44, 0.20)
    mod = np.zeros(X.shape)
    for kx, ky, ph, amp in world.ground_waves:
        mod += amp * np.sin(kx * X + ky * Y + ph)
    base *= (1.0 + 2.5 * mod)[..., None]

    for cx, cy, w, l, yaw, period in world.fields:
        c, s = np.cos(yaw), np.sin(yaw)
        lx = (X - cx) * c + (Y - cy) * s
        ly = -(X - cx) * s + (Y - cy) * c
        inside = (np.abs(lx) < w / 2) & (np.abs(ly) < l / 2)
        stripe = 0.5 + 0.5 * np.sin(2 * np.pi * lx / period)
        col = np.stack([0.55 + 0.10 * stripe, 0.48 + 0.06 * stripe, 0.25 + 0.04 * stripe], -1)
        base[inside] = col[inside]

    road = _dist_to_roads(X, Y, world.roads) < world.road_width / 2
    base[road] = (0.42, 0.42, 0.43)
    return np.clip(base, 0, 1)


def is_road(world: World, X, Y) -> np.ndarray:
    return _dist_to_roads(X, Y, world.roads) < world.road_width / 2


def object_masks(world: World, idx: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Boolean (K, *X.shape) footprint masks for the objects in idx."""
    o = world.obj
    cx, cy = o["cx"][idx], o["cy"][idx]
    shp = (len(idx),) + (1,) * X.ndim
    dx = X[None] - cx.reshape(shp)
    dy = Y[None] - cy.reshape(shp)
    circ = np.isin(o["kind"][idx], CIRCULAR).reshape(shp)
    c, s = np.cos(o["yaw"][idx]).reshape(shp), np.sin(o["yaw"][idx]).reshape(shp)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    w, l = o["w"][idx].reshape(shp), o["l"][idx].reshape(shp)
    rect_in = (np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)  # length along yaw
    circ_in = dx**2 + dy**2 <= (w / 2) ** 2
    return np.where(circ, circ_in, rect_in)
