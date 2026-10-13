"""Reading raw flight logs (the format the vehicles dump after a mission)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class RawFlight:
    root: Path
    meta: dict
    camera_ts: np.ndarray      # int64 microseconds, sorted
    lidar_ts: np.ndarray       # int64 microseconds (LiDAR clock), sorted
    odom: np.ndarray           # (N, 5): t_s, x, y, z, yaw
    labels: dict               # ts_us -> list of boxes

    @property
    def flight_id(self) -> str:
        return self.meta["flight_id"]

    def image(self, ts: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / "camera" / f"{ts}.png").convert("RGB"))

    def sweep(self, ts: int) -> np.ndarray:
        return np.fromfile(self.root / "lidar" / f"{ts}.bin", dtype=np.float32).reshape(-1, 4)

    def pose(self, t_s):
        """Interpolated (x, y, z, yaw) at time(s) t_s (seconds)."""
        o = self.odom
        return tuple(np.interp(t_s, o[:, 0], o[:, i]) for i in range(1, 5))


def _ts_list(folder: Path, suffix: str) -> np.ndarray:
    return np.array(sorted(int(p.stem) for p in folder.glob(f"*{suffix}")), dtype=np.int64)


def load_flight(root: Path) -> RawFlight:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    odom = np.loadtxt(root / "odometry.csv", delimiter=",", skiprows=1)
    odom[:, 0] /= 1e6
    labels = {}
    with open(root / "labels.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            labels[int(rec["ts_us"])] = rec["boxes"]
    return RawFlight(root, meta, _ts_list(root / "camera", ".png"),
                     _ts_list(root / "lidar", ".bin"), odom, labels)


def hash_tree(root: Path) -> str:
    """Content hash of a raw flight folder (file names + bytes), order independent."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def list_flights(raw_root: Path) -> list[Path]:
    return sorted(p for p in Path(raw_root).iterdir() if (p / "meta.json").exists())
