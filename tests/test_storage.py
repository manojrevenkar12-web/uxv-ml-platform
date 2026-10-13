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


def test_s3_backend_with_cache(tmp_path, monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    from uxvml.data.store import S3Store, open_store

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    with moto.mock_aws():
        boto3.client("s3").create_bucket(Bucket="uxv", CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        s3 = S3Store("uxv", "datasets")
        d = s3.put(b"shard-bytes")
        assert s3.exists(d) and s3.get(d) == b"shard-bytes"
        s3.put_ref("datasets/x/latest", {"version": "v1"})
        assert s3.has_ref("datasets/x/latest") and s3.get_ref("datasets/x/latest") == {"version": "v1"}
        assert s3.list_refs("datasets/") == ["datasets/x/latest"]

        monkeypatch.setenv("UXVML_CACHE_DIR", str(tmp_path / "cache"))
        cached = open_store("s3://uxv/datasets")
        assert isinstance(cached, CachedStore)
        assert cached.get(d) == b"shard-bytes" and cached.get(d) == b"shard-bytes"
        assert (cached.misses, cached.hits) == (1, 1)
        assert cached.get_ref("datasets/x/latest")["version"] == "v1"
