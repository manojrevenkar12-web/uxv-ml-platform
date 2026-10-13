"""Heatmap targets, loss, decoding and detection metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import N_CLASSES, STRIDE

GRID = 96 // STRIDE


def make_targets(boxes: np.ndarray, grid: int = GRID):
    """boxes (N,5) cls,x,y,w,h in pixels -> heatmap (C,G,G), reg (4,G,G), mask (G,G)."""
    heat = np.zeros((N_CLASSES, grid, grid), np.float32)
    reg = np.zeros((4, grid, grid), np.float32)
    mask = np.zeros((grid, grid), np.float32)
    ys, xs = np.mgrid[0:grid, 0:grid]
    for c, x, y, w, h in boxes:
        c = int(c)
        gx, gy = x / STRIDE, y / STRIDE
        ix, iy = min(int(gx), grid - 1), min(int(gy), grid - 1)
        sigma = max(0.6, 0.12 * np.sqrt(max(w, 0.5) * max(h, 0.5)) / STRIDE * 2)
        g = np.exp(-(((xs + 0.5) - gx) ** 2 + ((ys + 0.5) - gy) ** 2) / (2 * sigma**2))
        heat[c] = np.maximum(heat[c], g)
        heat[c, iy, ix] = 1.0
        reg[:, iy, ix] = (np.log(max(w, 0.5)), np.log(max(h, 0.5)), gx - ix, gy - iy)
        mask[iy, ix] = 1.0
    return heat, reg, mask


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def detection_loss(heat_logits, reg_out, heat_t, reg_t, mask):
    lh = focal_loss(heat_logits, heat_t)
    m = mask.unsqueeze(1)
    n = mask.sum().clamp(min=1)
    l_size = (F.l1_loss(reg_out[:, :2], reg_t[:, :2], reduction="none") * m).sum() / n
    l_off = (F.l1_loss(reg_out[:, 2:], reg_t[:, 2:], reduction="none") * m).sum() / n
    return lh + 0.2 * l_size + 1.0 * l_off, {"heat": lh.item(), "size": l_size.item(), "off": l_off.item()}


@torch.no_grad()
def decode(heat_logits, reg_out, k=60, min_score=0.02):
    """-> list (per image) of arrays (M,6): cls, x, y, w, h, score."""
    heat = torch.sigmoid(heat_logits)
    peaks = heat * (F.max_pool2d(heat, 3, 1, 1) == heat).float()
    B, C, G, _ = heat.shape
    scores, idx = peaks.reshape(B, -1).topk(min(k, C * G * G))
    out = []
    for b in range(B):
        s, i = scores[b], idx[b]
        keep = s >= min_score
        s, i = s[keep], i[keep]
        c = i // (G * G)
        cell = i % (G * G)
        iy, ix = cell // G, cell % G
        r = reg_out[b, :, iy, ix]
        x = (ix.float() + r[2]) * STRIDE
        y = (iy.float() + r[3]) * STRIDE
        w, h = torch.exp(r[0]), torch.exp(r[1])
        out.append(torch.stack([c.float(), x, y, w, h, s], 1).cpu().numpy())
    return out


def match_tolerance(w, h):
    return max(2.0, 0.35 * min(w, h))


def match(preds: np.ndarray, gts: np.ndarray):
    """Greedy centre-distance matching. Returns (pred_is_tp, gt_matched_idx_per_pred)."""
    order = np.argsort(-preds[:, 5]) if len(preds) else np.array([], int)
    used = np.zeros(len(gts), bool)
    tp = np.zeros(len(preds), bool)
    assign = np.full(len(preds), -1)
    for i in order:
        c, x, y = preds[i, 0], preds[i, 1], preds[i, 2]
        best, bd = -1, np.inf
        for j in range(len(gts)):
            if used[j] or gts[j, 0] != c:
                continue
            d = np.hypot(gts[j, 1] - x, gts[j, 2] - y)
            if d <= match_tolerance(gts[j, 3], gts[j, 4]) and d < bd:
                best, bd = j, d
        if best >= 0:
            used[best] = True
            tp[i] = True
            assign[i] = best
    return tp, assign


def average_precision(scores, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, float)[o]
    ctp = np.cumsum(tp)
    prec = ctp / np.arange(1, len(tp) + 1)
    rec = ctp / n_gt
    mprec = np.concatenate([[0], prec, [0]])
    mrec = np.concatenate([[0], rec, [rec[-1]]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mprec[1:]))


def per_class_records(preds_list, gts_list):
    """Collect (score, tp) per class and GT counts, for AP and bootstrap."""
    recs = []
    for preds, gts in zip(preds_list, gts_list):
        tp, _ = match(preds, gts)
        recs.append({
            "scores": preds[:, 5],
            "cls": preds[:, 0].astype(int),
            "tp": tp,
            "n_gt": np.bincount(gts[:, 0].astype(int), minlength=N_CLASSES) if len(gts) else np.zeros(N_CLASSES, int),
        })
    return recs


def map_from_records(recs, classes=range(N_CLASSES)):
    aps = {}
    for c in classes:
        s, t, n = [], [], 0
        for r in recs:
            m = r["cls"] == c
            s.append(r["scores"][m])
            t.append(r["tp"][m])
            n += int(r["n_gt"][c])
        aps[c] = average_precision(np.concatenate(s) if s else [], np.concatenate(t) if t else [], n)
    vals = [v for v in aps.values() if not np.isnan(v)]
    return aps, float(np.mean(vals)) if vals else float("nan")
