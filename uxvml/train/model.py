"""Small centre-point detector (CenterNet-style) for camera or camera+LiDAR input.

Output stride 2 (48x48 heatmaps for 96x96 input) so that people, which are
only 1-3 pixels wide from survey altitude, still get their own cell.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 3
STRIDE = 2


def conv_bn(cin, cout, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CenterNetLite(nn.Module):
    def __init__(self, in_ch: int = 5, width: int = 32, n_classes: int = N_CLASSES):
        super().__init__()
        w = width
        # space-to-depth keeps every input pixel (small objects survive) while the
        # network itself runs at 48x48 - about 4x cheaper than a full-res stem
        self.stem = nn.Sequential(nn.PixelUnshuffle(2), conv_bn(4 * in_ch, w), conv_bn(w, w))   # 48
        self.down2 = nn.Sequential(conv_bn(w, 2 * w, 2), conv_bn(2 * w, 2 * w),
                                   conv_bn(2 * w, 2 * w, dilation=2))                         # 24
        self.down3 = nn.Sequential(conv_bn(2 * w, 2 * w, 2), conv_bn(2 * w, 2 * w))          # 12
        self.lat2 = nn.Conv2d(2 * w, w, 1)
        self.lat3 = nn.Conv2d(2 * w, w, 1)
        self.fuse = conv_bn(w, w)
        self.heat = nn.Sequential(conv_bn(w, w), nn.Conv2d(w, n_classes, 1))
        self.reg = nn.Sequential(conv_bn(w, w), nn.Conv2d(w, 4, 1))       # log w, log h, dx, dy
        nn.init.constant_(self.heat[-1].bias, -2.2)

    def features(self, x):
        d1 = self.stem(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        p = d1 + F.interpolate(self.lat2(d2) + F.interpolate(self.lat3(d3), scale_factor=2), scale_factor=2)
        return self.fuse(p), d3

    def forward(self, x):
        p, _ = self.features(x)
        return self.heat(p), self.reg(p)

    @torch.no_grad()
    def embed(self, x):
        """Global descriptor for similarity search: pooled deep features."""
        p, d3 = self.features(x)
        e = torch.cat([F.adaptive_avg_pool2d(d3, 1).flatten(1), F.adaptive_max_pool2d(p, 1).flatten(1)], 1)
        return F.normalize(e, dim=1)
