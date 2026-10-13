"""Content-addressed object store with named refs.

Objects (shards, indexes, reports) are stored by SHA-256 of their bytes, so a
digest in a manifest pins exact content and identical data is never stored
twice. Refs are small mutable JSON documents (dataset versions, "latest"
pointers). Backends: local filesystem, or any S3-compatible bucket (AWS S3,
MinIO, Ceph) when boto3 is installed. A size-capped local read cache sits in
front of a remote store for training jobs.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "refs").mkdir(parents=True, exist_ok=True)

    def _obj(self, d: str) -> Path:
        return self.root / "objects" / d[:2] / d[2:]

    def put(self, data: bytes) -> str:
        d = digest(data)
        p = self._obj(d)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=p.parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)  # atomic: readers never see partial objects
        return d

    def get(self, d: str) -> bytes:
        data = self._obj(d).read_bytes()
        if digest(data) != d:
            raise IOError(f"object {d[:12]} is corrupt (hash mismatch)")
        return data

    def exists(self, d: str) -> bool:
        return self._obj(d).exists()

    def put_ref(self, name: str, doc: dict) -> None:
        p = self.root / "refs" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        os.replace(tmp, p)

    def get_ref(self, name: str) -> dict:
        return json.loads((self.root / "refs" / name).read_text())

    def has_ref(self, name: str) -> bool:
        return (self.root / "refs" / name).exists()

    def list_refs(self, prefix: str = "") -> list[str]:
        base = self.root / "refs"
        return sorted(str(p.relative_to(base)) for p in base.rglob("*")
                      if p.is_file() and str(p.relative_to(base)).startswith(prefix))


class S3Store:
    """S3-compatible backend (set AWS_ENDPOINT_URL for MinIO / on-prem object storage)."""

    def __init__(self, bucket: str, prefix: str = ""):
        import boto3  # optional dependency

        self.s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
        self.bucket, self.prefix = bucket, prefix.strip("/")

    def _key(self, *parts):
        return "/".join(p for p in (self.prefix, *parts) if p)

    def put(self, data: bytes) -> str:
        d = digest(data)
        if not self.exists(d):
            self.s3.put_object(Bucket=self.bucket, Key=self._key("objects", d[:2], d[2:]), Body=data)
        return d

    def get(self, d: str) -> bytes:
        data = self.s3.get_object(Bucket=self.bucket, Key=self._key("objects", d[:2], d[2:]))["Body"].read()
        if digest(data) != d:
            raise IOError(f"object {d[:12]} is corrupt (hash mismatch)")
        return data

    def exists(self, d: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key("objects", d[:2], d[2:]))
            return True
        except Exception:
            return False

    def put_ref(self, name, doc):
        self.s3.put_object(Bucket=self.bucket, Key=self._key("refs", name),
                           Body=json.dumps(doc, indent=2, sort_keys=True).encode())

    def get_ref(self, name):
        return json.loads(self.s3.get_object(Bucket=self.bucket, Key=self._key("refs", name))["Body"].read())

    def has_ref(self, name):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key("refs", name))
            return True
        except Exception:
            return False

    def list_refs(self, prefix=""):
        out, token = [], None
        base = self._key("refs") + "/"
        while True:
            kw = {"Bucket": self.bucket, "Prefix": base + prefix}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [o["Key"][len(base):] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return sorted(out)
            token = r["NextContinuationToken"]


class CachedStore:
    """Read-through local cache (LRU by access time, size-capped) in front of a remote store."""

    def __init__(self, remote, cache_dir: str | Path, max_bytes: int = 20 * 2**30):
        self.remote = remote
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.hits = self.misses = 0

    def get(self, d: str) -> bytes:
        p = self.cache / d
        if p.exists():
            self.hits += 1
            os.utime(p)
            return p.read_bytes()
        self.misses += 1
        data = self.remote.get(d)
        fd, tmp = tempfile.mkstemp(dir=self.cache)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        self._evict()
        return data

    def _evict(self):
        files = [(p.stat().st_atime, p.stat().st_size, p) for p in self.cache.iterdir() if p.is_file()]
        total = sum(s for _, s, _ in files)
        for _, size, p in sorted(files):
            if total <= self.max_bytes:
                break
            p.unlink(missing_ok=True)
            total -= size

    def __getattr__(self, name):  # put / refs go straight to the remote
        return getattr(self.remote, name)


def open_store(uri: str | Path):
    uri = str(uri)
    if uri.startswith("s3://"):
        bucket, _, prefix = uri[5:].partition("/")
        store = S3Store(bucket, prefix)
        cache = os.environ.get("UXVML_CACHE_DIR")
        return CachedStore(store, cache) if cache else store
    if uri.startswith("file://"):
        uri = uri[7:]
    return LocalStore(uri)
