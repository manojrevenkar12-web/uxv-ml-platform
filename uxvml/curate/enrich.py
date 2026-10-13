"""Automated metadata generation for every frame of a dataset version.

* Image-physics features (luminance, dark-channel haze estimate, noise level,
  contrast, saturation) -> lighting and fog tags. The taggers are tiny
  classifiers calibrated on a handful of hand-tagged flights.
* LiDAR features -> built-up fraction, max structure height, point density.
* Detector outputs -> predicted object counts and a learned embedding per frame,
  used for query-by-example search and hard-example mining.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..data.loader import ShardStream, collate
from ..data.shardio import pack, unpack
from ..sim.world import CLASSES
from ..train.targets import decode

FEATURES = ["lum", "dark_p10", "dark_ratio", "contrast", "noise", "saturation",
            "lidar_density", "built_up", "max_height"]
COUNT_THRESHOLD = 0.35


def image_features(img: np.ndarray) -> dict:
    x = img.astype(np.float32) / 255.0
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = sliding_window_view(x.min(-1), (7, 7)).min((-1, -2))
    lap = lum[1:-1, 1:-1] * 4 - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2] - lum[1:-1, 2:]
    mx, mn = x.max(-1), x.min(-1)
    return {
        "lum": float(lum.mean()),
        "dark_p10": float(np.percentile(dark, 10)),
        "dark_ratio": float(np.percentile(dark, 10) / (lum.mean() + 1e-3)),
        "contrast": float(lum.std()),
        "noise": float(np.median(np.abs(lap)) * 1.4826 / np.sqrt(20)),
        "saturation": float(((mx - mn) / (mx + 1e-3)).mean()),
    }


def lidar_features(lid: np.ndarray) -> dict:
    h, c = lid[0].astype(np.float32), lid[1].astype(np.float32)
    return {"lidar_density": float((c > 0).mean()), "built_up": float((h > 3.0).mean()),
            "max_height": float(h.max())}


@torch.no_grad()
def model_outputs(model, store, manifest, inputs: str, split: str):
    model.eval()
    ds = ShardStream(store, manifest, split, inputs=inputs, train=False)
    rows, embs = [], []
    for b in DataLoader(ds, batch_size=64, collate_fn=collate):
        x = b["x"].contiguous(memory_format=torch.channels_last)
        heat, reg = model(x)
        dets = decode(heat, reg)
        e = model.embed(x).numpy().astype(np.float16)
        for sid, d, ee in zip(b["sid"], dets, e):
            keep = d[d[:, 5] >= COUNT_THRESHOLD]
            cnt = np.bincount(keep[:, 0].astype(int), minlength=len(CLASSES))
            rows.append({"sample_id": sid, **{f"pred_{c}": int(cnt[i]) for i, c in enumerate(CLASSES)}})
            embs.append(ee)
    return pd.DataFrame(rows), np.stack(embs)


def raw_features(store, manifest) -> pd.DataFrame:
    rows = []
    for s in manifest["shard_info"]:
        sh = unpack(store.get(s["digest"]))
        for r, sid in enumerate(sh["sample_ids"]):
            rows.append({"sample_id": sid.decode(), "split": s["split"],
                         **image_features(sh["image"][r]), **lidar_features(sh["lidar"][r])})
    return pd.DataFrame(rows)


class _Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=object)


def fit_tagger(df: pd.DataFrame, labels: pd.Series):
    if labels.nunique() < 2:  # calibration set shows only one condition
        return _Constant(labels.iloc[0])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(df[FEATURES].values, labels.values)
    return clf


def enrich(store, manifest, model, inputs: str, calibration: pd.DataFrame | None = None):
    """Returns (frame table, embeddings, sample order). calibration: sample_id, lighting, fog."""
    feats = raw_features(store, manifest)
    if calibration is not None and len(calibration):
        cal = feats.merge(calibration, on="sample_id")
        light_clf = fit_tagger(cal, cal["lighting"])
        fog_clf = fit_tagger(cal, cal["fog"])
        feats["lighting"] = light_clf.predict(feats[FEATURES].values)
        feats["fog"] = fog_clf.predict(feats[FEATURES].values)
        feats["tag_source"] = np.where(feats.sample_id.isin(cal.sample_id), "human", "auto")
    preds, embs, order = [], [], []
    for split in ("train", "val", "test"):
        p, e = model_outputs(model, store, manifest, inputs, split)
        preds.append(p)
        embs.append(e)
    preds = pd.concat(preds, ignore_index=True)
    table = feats.merge(preds, on="sample_id", how="left")
    emb = np.concatenate(embs)
    emb_order = preds.sample_id.tolist()
    return table, emb, emb_order


def save_enrichment(store, manifest, table: pd.DataFrame, emb: np.ndarray, order: list, model_ref: str):
    buf = io.BytesIO()
    table.to_parquet(buf, index=False)
    t_digest = store.put(buf.getvalue())
    e_digest = store.put(pack({"emb": emb, "sample_ids": np.array(order).astype("S48")}))
    doc = {"dataset_version": manifest["version"], "table": t_digest, "embeddings": e_digest,
           "model": model_ref}
    store.put_ref(f"enrichment/{manifest['name']}/{manifest['version']}", doc)
    return doc


def load_enrichment(store, manifest):
    doc = store.get_ref(f"enrichment/{manifest['name']}/{manifest['version']}")
    table = pd.read_parquet(io.BytesIO(store.get(doc["table"])))
    e = unpack(store.get(doc["embeddings"]))
    return table, np.asarray(e["emb"], np.float32), [s.decode() for s in e["sample_ids"]]
