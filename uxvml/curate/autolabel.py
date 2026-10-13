"""Model-assisted labelling and label-error detection.

Auto-labelling: per-class confidence thresholds are calibrated on the
validation split to hit a target precision. A frame is auto-labelled only if
every detection is clearly confident or clearly background; frames with any
borderline detection go to a human review queue.

Label audit: a model trained on trusted labels is run over other annotation
batches; disagreements become a ranked list of suspect labels for review.
"""
from __future__ import annotations

import numpy as np

from ..sim.world import CLASSES
from ..train.targets import match, match_tolerance
from ..train.trainer import predict


def calibrate_thresholds(model, store, manifest, inputs, target_precision=0.95, min_support=30):
    _, preds, gts = predict(model, store, manifest, "val", inputs)
    out = {}
    for c in range(len(CLASSES)):
        scores, tps = [], []
        for p, g in zip(preds, gts):
            tp, _ = match(p, g)
            m = p[:, 0] == c
            scores.append(p[m, 5])
            tps.append(tp[m])
        s, t = np.concatenate(scores), np.concatenate(tps)
        o = np.argsort(-s)
        s, t = s[o], t[o]
        prec = np.cumsum(t) / np.arange(1, len(t) + 1)
        ok = np.nonzero((prec >= target_precision) & (np.arange(1, len(t) + 1) >= min_support))[0]
        # lowest threshold whose cumulative precision still meets the target
        out[CLASSES[c]] = float(s[ok[-1]]) if len(ok) else 1.01
    return out


def autolabel(model, store, manifest, inputs, sample_ids: set, thresholds: dict,
              review_band: float = 0.4, split="train"):
    sids, preds, _ = predict(model, store, manifest, split, inputs, include=sample_ids)
    tau = np.array([thresholds[c] for c in CLASSES])
    labels, review = {}, []
    for sid, p in zip(sids, preds):
        t = tau[p[:, 0].astype(int)]
        accepted = p[p[:, 5] >= t]
        uncertain = (p[:, 5] < t) & (p[:, 5] >= t * review_band)
        if uncertain.any():
            review.append(sid)
        else:
            labels[sid] = accepted[:, :5].astype(np.float32)
    return labels, review


def audit_labels(model, store, manifest, inputs, sample_ids: set, t_high=0.6, t_low=0.15, split="train"):
    sids, preds, gts = predict(model, store, manifest, split, inputs, include=sample_ids)
    flags = []
    for sid, p, g in zip(sids, preds, gts):
        p = p[p[:, 5] >= t_low]
        for j, gb in enumerate(g):
            tol = match_tolerance(gb[3], gb[4])
            d = np.hypot(p[:, 1] - gb[1], p[:, 2] - gb[2]) if len(p) else np.array([])
            near = p[d <= tol] if len(p) else p
            if len(near) == 0:
                flags.append({"sample_id": sid, "kind": "unsupported_box", "box": j, "score": 1.0})
                continue
            same = near[near[:, 0] == gb[0]]
            if len(same) == 0:
                best = near[near[:, 5].argmax()]
                if best[5] >= t_high:
                    flags.append({"sample_id": sid, "kind": "class_mismatch", "box": j,
                                  "score": float(best[5]), "predicted": CLASSES[int(best[0])]})
        for pb in p[p[:, 5] >= t_high]:
            if not len(g):
                covered = False
            else:
                tol = np.array([match_tolerance(b[3], b[4]) for b in g])
                covered = (np.hypot(g[:, 1] - pb[1], g[:, 2] - pb[2]) <= np.maximum(tol, 0.5 * max(pb[3], pb[4]))).any()
            if not covered:
                flags.append({"sample_id": sid, "kind": "missing_label", "box": None,
                              "score": float(pb[5]), "pred": [round(float(v), 2) for v in pb[:5]]})
    return flags
