"""Raw flight logs -> validated, sharded, versioned dataset.

A dataset version is identified by a hash over everything that determines its
content: raw input hashes, ingestion parameters, the ingestion code version,
and the resulting shard/index digests. Re-running ingestion on the same inputs
gives the same version id; changing any input gives a new one.
"""
from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..sim.world import CLASSES
from .quality import run_flight_qc
from .raw import hash_tree, list_flights, load_flight
from .shardio import pack
from .sync import fusion_raster, gsd

INGEST_VERSION = "1.2.0"   # bump when ingestion logic changes output


@dataclass
class IngestParams:
    shard_size: int = 256
    sync_correction: bool = True
    drop_camera_faults: bool = True
    n_val_per_tod: int = 2
    n_test_per_tod: int = 2
    trusted_vendors: tuple = ("vendor-b",)   # only these may label val/test


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _ingest_flight(args):
    path, params = args
    flight = load_flight(Path(path))
    qc = run_flight_qc(flight, estimate_sync=params.sync_correction)
    offset = qc["sync"]["applied_offset_s"] if params.sync_correction else 0.0
    lidar_ts_s = flight.lidar_ts / 1e6 - offset
    odo = flight.odom
    speed = np.hypot(np.gradient(odo[:, 1], odo[:, 0]), np.gradient(odo[:, 2], odo[:, 0]))
    samples = []
    for ts in flight.camera_ts:
        ts = int(ts)
        fq = qc["frames"][ts]
        if params.drop_camera_faults and fq["camera_fault"]:
            continue
        if not fq["has_labels"]:
            continue
        raster, sweep_ts, dt = fusion_raster(flight, ts, offset, lidar_ts_s)
        x, y, z, yaw = flight.pose(ts / 1e6)
        boxes = flight.labels[ts]
        arr = np.array([[b["cls"], b["x"], b["y"], b["w"], b["h"]] for b in boxes],
                       dtype=np.float32).reshape(-1, 5)
        counts = np.bincount(arr[:, 0].astype(int), minlength=len(CLASSES)) if len(arr) else np.zeros(len(CLASSES), int)
        samples.append({
            "image": flight.image(ts),
            "lidar": raster.astype(np.float16),
            "boxes": arr,
            "meta": {
                "sample_id": f"{flight.flight_id}/{ts}",
                "flight_id": flight.flight_id, "ts_us": ts,
                "altitude_m": round(float(z), 2), "gsd_m": round(float(gsd(z)), 4),
                "speed_mps": round(float(np.interp(ts / 1e6, odo[:, 0], speed)), 2),
                "yaw": round(float(yaw), 4),
                "brightness": fq["brightness"], "contrast": fq["contrast"],
                "lidar_ok": sweep_ts is not None,
                "lidar_sweep_ts_us": sweep_ts if sweep_ts is not None else -1,
                "lidar_dt_ms": round(dt * 1000, 1) if np.isfinite(dt) else -1.0,
                "n_boxes": int(len(arr)),
                **{f"n_{c}": int(counts[i]) for i, c in enumerate(CLASSES)},
                "suspect_boxes": int(sum(1 for i in fq["box_issues"] if i)),
                "vendor": flight.meta.get("annotation_vendor", "unknown"),
            },
        })
    qc_small = {k: v for k, v in qc.items() if k != "frames"}
    qc_small["frame_faults"] = {str(t): f["camera_fault"] for t, f in qc["frames"].items() if f["camera_fault"]}
    qc_small["suspect_boxes"] = {str(t): [i for i, iss in enumerate(f["box_issues"]) if iss]
                                 for t, f in qc["frames"].items() if any(f["box_issues"])}
    return flight.flight_id, flight.meta, hash_tree(Path(path)), qc_small, samples


def assign_splits(flight_meta: dict, qc: dict, samples_by_flight: dict, params: IngestParams) -> dict:
    """Flight-level splits (no leakage between frames of one flight).

    Val/test only use trusted annotation vendors, stratified by lighting
    (median frame brightness bucket) so every condition is represented.
    """
    def bucket(fid):
        b = np.median([s["meta"]["brightness"] for s in samples_by_flight[fid]] or [0])
        return "dark" if b < 40 else ("dim" if b < 85 else "bright")

    def key(fid):
        return hashlib.sha256(fid.encode()).hexdigest()

    split = {}
    trusted = [f for f in flight_meta if flight_meta[f].get("annotation_vendor") in params.trusted_vendors]
    for f in flight_meta:
        split[f] = "train"
    groups: dict[str, list] = {}
    for f in trusted:
        groups.setdefault(bucket(f), []).append(f)
    for g in groups.values():
        g.sort(key=key)
        for f in g[:params.n_test_per_tod]:
            split[f] = "test"
        for f in g[params.n_test_per_tod:params.n_test_per_tod + params.n_val_per_tod]:
            split[f] = "val"
    return split


def build_dataset(raw_root: Path, store, name: str, params: IngestParams | None = None,
                  workers: int = 2, log=print) -> dict:
    params = params or IngestParams()
    paths = list_flights(raw_root)
    log(f"ingesting {len(paths)} flights with {workers} workers")
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_ingest_flight, [(str(p), params) for p in paths]))

    flight_meta, raw_hashes, qcs, samples_by_flight = {}, {}, {}, {}
    for fid, meta, rh, qc, samples in results:
        flight_meta[fid], raw_hashes[fid], qcs[fid], samples_by_flight[fid] = meta, rh, qc, samples
    split = assign_splits(flight_meta, qcs, samples_by_flight, params)

    shards, rows = [], []
    for sp in ("train", "val", "test"):
        pool = [s for fid in sorted(samples_by_flight) if split[fid] == sp for s in samples_by_flight[fid]]
        for i in range(0, len(pool), params.shard_size):
            chunk = pool[i:i + params.shard_size]
            offsets = np.cumsum([0] + [len(s["boxes"]) for s in chunk]).astype(np.int64)
            data = pack({
                "image": np.stack([s["image"] for s in chunk]),
                "lidar": np.stack([s["lidar"] for s in chunk]),
                "boxes": np.concatenate([s["boxes"] for s in chunk]) if offsets[-1] else np.zeros((0, 5), np.float32),
                "box_offsets": offsets,
                "sample_ids": np.array([s["meta"]["sample_id"] for s in chunk]).astype("S48"),
            })
            d = store.put(data)
            shards.append({"digest": d, "split": sp, "samples": len(chunk), "bytes": len(data)})
            for r, s in enumerate(chunk):
                rows.append({**s["meta"], "split": sp, "shard": d, "row": r})

    table = pa.Table.from_pylist(rows)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    index_digest = store.put(buf.getvalue())
    qc_digest = store.put(canonical(qcs))

    core = {
        "name": name,
        "ingest_version": INGEST_VERSION,
        "params": asdict(params),
        "inputs": {f: raw_hashes[f] for f in sorted(raw_hashes)},
        "shards": [s["digest"] for s in shards],
        "index": index_digest,
    }
    version = hashlib.sha256(canonical(core)).hexdigest()[:16]
    parent = store.get_ref(f"datasets/{name}/latest")["version"] if store.has_ref(f"datasets/{name}/latest") else None
    split_counts = {sp: sum(s["samples"] for s in shards if s["split"] == sp) for sp in ("train", "val", "test")}
    manifest = {
        **core,
        "version": version,
        "parent": parent if parent != version else None,
        "splits": {f: split[f] for f in sorted(split)},
        "shard_info": shards,
        "qc_report": qc_digest,
        "stats": {
            "flights": len(paths),
            "samples": len(rows),
            "split_samples": split_counts,
            "bytes": sum(s["bytes"] for s in shards),
            "camera_faults_dropped": sum(q["summary"]["camera_faults"] for q in qcs.values()),
            "suspect_boxes": sum(q["summary"]["suspect_boxes"] for q in qcs.values()),
            "lidar_missing_frames": sum(1 for r in rows if not r["lidar_ok"]),
            "sync_corrected_flights": sorted(f for f, q in qcs.items() if q["sync"]["status"] == "corrected"),
            "sync_unverified_flights": sorted(f for f, q in qcs.items() if q["sync"]["status"] == "unverified"),
        },
    }
    store.put_ref(f"datasets/{name}/{version}.json", manifest)
    store.put_ref(f"datasets/{name}/latest", {"version": version})
    log(f"dataset {name}@{version}: {len(rows)} samples in {len(shards)} shards {split_counts}")
    return manifest


def load_manifest(store, name: str, version: str = "latest") -> dict:
    if version == "latest":
        version = store.get_ref(f"datasets/{name}/latest")["version"]
    return store.get_ref(f"datasets/{name}/{version}.json")


def load_index(store, manifest: dict):
    return pq.read_table(io.BytesIO(store.get(manifest["index"]))).to_pandas()
