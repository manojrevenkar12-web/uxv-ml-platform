"""Deterministic binary shard format.

layout:  b"UXS1" | uint64 header length | JSON header | raw array bytes
The header lists each array's dtype, shape and byte offset. Unlike zip-based
.npz files there are no embedded timestamps, so the same samples always
produce the same bytes (and the same content hash). Arrays can be read with
zero copies via np.frombuffer.
"""
from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"UXS1"


def pack(arrays: dict[str, np.ndarray]) -> bytes:
    header, chunks, offset = {}, [], 0
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        header[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "offset": offset}
        b = a.tobytes()
        chunks.append(b)
        offset += len(b)
    h = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<Q", len(h)) + h + b"".join(chunks)


def unpack(data: bytes) -> dict[str, np.ndarray]:
    if data[:4] != MAGIC:
        raise ValueError("not a UXS1 shard")
    (hlen,) = struct.unpack("<Q", data[4:12])
    header = json.loads(data[12:12 + hlen])
    base = 12 + hlen
    out = {}
    for name, h in header.items():
        n = int(np.prod(h["shape"])) if h["shape"] else 1
        out[name] = np.frombuffer(data, dtype=np.dtype(h["dtype"]), count=n,
                                  offset=base + h["offset"]).reshape(h["shape"])
    return out
