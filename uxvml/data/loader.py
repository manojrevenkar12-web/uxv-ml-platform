"""High-throughput streaming over dataset shards.

Shards (not samples) are the unit of distribution: each shard is assigned to
exactly one (DDP rank, dataloader worker) per epoch, read with one sequential
request and decoded zero-copy. A shuffle buffer mixes samples across shards.
Training streams are infinite so every rank runs the same number of steps.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .shardio import unpack


def decode_sample(sh: dict, r: int, inputs: str):
    img = sh["image"][r].astype(np.float32) / 255.0                       # (96,96,3)
    x = [img.transpose(2, 0, 1)]
    if "lidar" in inputs:
        lid = sh["lidar"][r].astype(np.float32)                           # (2,48,48)
        lid = lid.repeat(2, axis=1).repeat(2, axis=2)
        x.append(np.stack([lid[0] / 10.0, lid[1] / 3.0]))
    o = sh["box_offsets"]
    boxes = np.array(sh["boxes"][o[r]:o[r + 1]], dtype=np.float32)
    return np.concatenate(x).astype(np.float32), boxes


def augment(x: np.ndarray, boxes: np.ndarray, rng: np.random.Generator):
    """Random 90-degree rotation + flip (nadir imagery has no canonical 'up')."""
    S = x.shape[-1]
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    b = boxes.copy()
    for _ in range(k):  # rotate 90 deg counter-clockwise
        x = np.rot90(x, 1, axes=(1, 2))
        bx, by = b[:, 1].copy(), b[:, 2].copy()
        b[:, 1], b[:, 2] = by, S - bx
        b[:, 3], b[:, 4] = b[:, 4].copy(), b[:, 3].copy()
    if flip:
        x = x[:, :, ::-1]
        b[:, 1] = S - b[:, 1]
    return np.ascontiguousarray(x), b


class ShardStream(IterableDataset):
    def __init__(self, store, manifest: dict, split: str, inputs: str = "rgb+lidar",
                 train: bool = True, seed: int = 0, rank: int = 0, world: int = 1,
                 buffer: int = 256, include: set | None = None,
                 label_override: dict | None = None, target_fn=None):
        self.store, self.manifest, self.split = store, manifest, split
        self.inputs, self.train, self.seed = inputs, train, seed
        self.rank, self.world, self.buffer = rank, world, buffer
        self.include, self.label_override = include, label_override
        self.target_fn = target_fn
        self.shards = [s["digest"] for s in manifest["shard_info"] if s["split"] == split]
        self.bytes_read = 0

    def _my_shards(self, epoch):
        shards = list(self.shards)
        if self.train:
            rng = np.random.default_rng((self.seed, epoch))
            shards = [shards[i] for i in rng.permutation(len(shards))]
            shards = shards[self.rank::self.world]
        wi = get_worker_info()
        if wi is not None:
            shards = shards[wi.id::wi.num_workers]
        return shards

    def _samples(self, epoch):
        rng = np.random.default_rng((self.seed, epoch, self.rank, 7))
        for d in self._my_shards(epoch):
            data = self.store.get(d)
            self.bytes_read += len(data)
            sh = unpack(data)
            ids = [s.decode() for s in sh["sample_ids"]]
            for r, sid in enumerate(ids):
                if self.include is not None and sid not in self.include:
                    continue
                x, boxes = decode_sample(sh, r, self.inputs)
                if self.label_override is not None and sid in self.label_override:
                    boxes = self.label_override[sid]
                if self.train:
                    x, boxes = augment(x, boxes, rng)
                yield sid, x, boxes

    def _emit(self, sid, x, boxes):
        out = {"sid": sid, "x": torch.from_numpy(x), "boxes": boxes}
        if self.target_fn is not None:
            h, r, m = self.target_fn(boxes)
            out.update(heat=torch.from_numpy(h), reg=torch.from_numpy(r), mask=torch.from_numpy(m))
        return out

    def __iter__(self):
        if not self.train:
            for sid, x, b in self._samples(0):
                yield self._emit(sid, x, b)
            return
        rng = np.random.default_rng((self.seed, self.rank, 99))
        epoch = 0
        buf = []
        while True:  # infinite: the trainer decides how many steps make an epoch
            for item in self._samples(epoch):
                buf.append(item)
                if len(buf) >= self.buffer:
                    yield self._emit(*buf.pop(int(rng.integers(len(buf)))))
            epoch += 1


def collate(batch):
    out = {"sid": [b["sid"] for b in batch], "x": torch.stack([b["x"] for b in batch]),
           "boxes": [b["boxes"] for b in batch]}
    for k in ("heat", "reg", "mask"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out
