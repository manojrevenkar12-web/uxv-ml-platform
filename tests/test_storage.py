import numpy as np
import pytest

from uxvml.data.shardio import pack, unpack
from uxvml.data.store import CachedStore, LocalStore, digest


def test_shard_roundtrip_and_determinism():
    arrays = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.array([b"x", b"yy"]).astype("S4"),
              "c": np.zeros((0, 5), np.float32)}
    blob = pack(arrays)
    assert blob == pack(dict(reversed(list(arrays.items()))))  # key order does not matter
    back = unpack(blob)
    for k, v in arrays.items():
        np.testing.assert_array_equal(back[k], v)
        assert back[k].dtype == v.dtype


def test_unpack_rejects_foreign_bytes():
    with pytest.raises(ValueError):
        unpack(b"PK\x03\x04garbage")


def test_store_content_addressing_and_corruption(tmp_path):
    s = LocalStore(tmp_path)
    d = s.put(b"hello")
    assert d == digest(b"hello") and s.exists(d)
    assert s.put(b"hello") == d                      # idempotent
    assert s.get(d) == b"hello"
    s._obj(d).write_bytes(b"hellp")                 # bit rot
    with pytest.raises(IOError):
        s.get(d)


def test_refs(tmp_path):
    s = LocalStore(tmp_path)
    s.put_ref("datasets/x/v1.json", {"a": 1})
    s.put_ref("datasets/x/latest", {"version": "v1"})
    assert s.get_ref("datasets/x/v1.json") == {"a": 1}
    assert s.has_ref("datasets/x/latest") and not s.has_ref("datasets/y/latest")
    assert s.list_refs("datasets/x") == ["datasets/x/latest", "datasets/x/v1.json"]


def test_cached_store_hits_and_eviction(tmp_path):
    remote = LocalStore(tmp_path / "remote")
    digests = [remote.put(bytes([i]) * 1000) for i in range(5)]
    cache = CachedStore(remote, tmp_path / "cache", max_bytes=2500)
    for d in digests:
        cache.get(d)
    assert cache.misses == 5 and cache.hits == 0
    assert sum(p.stat().st_size for p in (tmp_path / "cache").iterdir()) <= 2500
    cache.get(digests[-1])
    assert cache.hits == 1
    assert cache.get(digests[0]) == bytes([0]) * 1000  # evicted entries are re-fetched
