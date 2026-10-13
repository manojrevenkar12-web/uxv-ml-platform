from dataclasses import replace

import numpy as np

from uxvml.data.dataset import build_dataset, load_index, load_manifest
from uxvml.data.loader import ShardStream, augment
from uxvml.data.shardio import unpack
from uxvml.data.store import open_store


def test_dataset_version_is_reproducible_and_input_sensitive(tiny_sim, tiny_dataset, tmp_path):
    sim, _ = tiny_sim
    _, _, manifest, params = tiny_dataset
    quiet = dict(workers=2, log=lambda *a: None)
    again = build_dataset(sim / "raw", open_store(tmp_path / "s1"), "tiny", params, **quiet)
    assert again["version"] == manifest["version"]
    assert again["shards"] == manifest["shards"]
    changed = build_dataset(sim / "raw", open_store(tmp_path / "s2"), "tiny", replace(params, shard_size=8), **quiet)
    assert changed["version"] != manifest["version"]


def test_manifest_lineage_and_split_hygiene(tiny_dataset):
    _, store, manifest, _ = tiny_dataset
    assert load_manifest(store, "tiny")["version"] == manifest["version"]
    assert set(manifest["inputs"]) == set(manifest["splits"])
    assert all(len(h) == 64 for h in manifest["inputs"].values())
    idx = load_index(store, manifest)
    per_flight = idx.groupby("flight_id").split.nunique()
    assert (per_flight == 1).all(), "a flight leaked across splits"
    assert set(idx.split) == {"train", "val", "test"}
    for s in manifest["shard_info"]:
        sh = unpack(store.get(s["digest"]))
        assert len(sh["sample_ids"]) == s["samples"]
        assert sh["box_offsets"][-1] == len(sh["boxes"])


def test_untrusted_vendor_never_in_eval_splits(tiny_sim, tmp_path):
    from uxvml.data.dataset import IngestParams
    sim, _ = tiny_sim
    m = build_dataset(sim / "raw", open_store(tmp_path), "t", IngestParams(n_val_per_tod=1, n_test_per_tod=1),
                      workers=2, log=lambda *a: None)
    idx = load_index(open_store(tmp_path), m)
    assert (idx[idx.split != "train"].vendor == "vendor-b").all()


def test_rank_sharding_is_disjoint_and_complete(tiny_dataset):
    _, store, manifest, _ = tiny_dataset
    seen = []
    for rank in range(2):
        ds = ShardStream(store, manifest, "train", rank=rank, world=2, seed=3)
        seen.append(set(ds._my_shards(epoch=0)))
    assert not (seen[0] & seen[1])
    assert seen[0] | seen[1] == set(ds.shards)


def test_eval_stream_covers_split_exactly_once(tiny_dataset):
    _, store, manifest, _ = tiny_dataset
    idx = load_index(store, manifest)
    ids = [x["sid"] for x in ShardStream(store, manifest, "val", train=False)]
    assert sorted(ids) == sorted(idx[idx.split == "val"].sample_id)


def test_augmentation_keeps_boxes_on_their_pixels():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.zeros((5, 96, 96), np.float32)
        x[:, 10:14, 60:70] = 1.0                       # rows 10-13, cols 60-69
        boxes = np.array([[0, 65.0, 12.0, 10.0, 4.0]], np.float32)
        xa, ba = augment(x, boxes, rng)
        ys, xs = np.nonzero(xa[0])
        assert abs((xs.min() + xs.max() + 1) / 2 - ba[0, 1]) < 1e-4
        assert abs((ys.min() + ys.max() + 1) / 2 - ba[0, 2]) < 1e-4
        assert abs(xs.max() + 1 - xs.min() - ba[0, 3]) < 1e-4
