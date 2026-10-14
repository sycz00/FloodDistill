"""Capacity-floor rungs (Stage 4). Opt-in; declared before any accuracy is seen.

All rungs map a normalised 6-band image (B,6,H,W) to two-class logits (B,2,H,W) at
input resolution, use ReLU6, and train from scratch. R1/R2 are strictly per-pixel
(1x1 convolutions only), so a pixel's prediction depends on its own six values only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _cbr(cin, cout, k=3, s=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class PerPixel(nn.Module):
    """R1 (hidden=()) = linear 6->2 (14 params); R2 (hidden=(16,16)) = MLP (418 params)."""
    def __init__(self, hidden=()):
        super().__init__()
        layers, c = [], 6
        for h in hidden:
            layers += [nn.Conv2d(c, h, 1), nn.ReLU6(inplace=True)]
            c = h
        layers.append(nn.Conv2d(c, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyConv(nn.Module):
    """R3: 3x3(6->8)-3x3(8->8)-1x1(8->2), BN+ReLU6, full resolution, RF 5 px."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(_cbr(6, 8), _cbr(8, 8), nn.Conv2d(8, 2, 1))

    def forward(self, x):
        return self.net(x)


class DWBlock(nn.Module):
    """Depthwise-separable residual block (MobileNet-style, no attention)."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = _cbr(cin, cin, 3, stride, groups=cin)
        self.pw = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return F.relu6(y + x if self.res else y)


class ContextConv(nn.Module):
    """R4: stride-2 stem, three stages (H/4, H/8, H/16), FPN-style skip decoder, no attention.

    Receptive field > 64 px (four stride-2 3x3 convs plus two stride-1 3x3 depthwise per
    stage). Decoder fuses H/16 -> H/8 -> H/4 with 1x1 projections, logits upsampled to H.
    """
    def __init__(self, widths=(12, 16, 24, 40), dec=16):  # 12,230 params (declared 8-14 k)
        super().__init__()
        c0, c1, c2, c3 = widths
        self.stem = _cbr(6, c0, 3, 2)
        self.s1 = nn.Sequential(DWBlock(c0, c1, 2), DWBlock(c1, c1, 1))
        self.s2 = nn.Sequential(DWBlock(c1, c2, 2), DWBlock(c2, c2, 1))
        self.s3 = nn.Sequential(DWBlock(c2, c3, 2), DWBlock(c3, c3, 1), DWBlock(c3, c3, 1))
        self.l3, self.l2, self.l1 = (nn.Conv2d(c, dec, 1) for c in (c3, c2, c1))
        self.fuse = _cbr(dec, dec, 3)
        self.cls = nn.Conv2d(dec, 2, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f1 = self.s1(self.stem(x)); f2 = self.s2(f1); f3 = self.s3(f2)
        y = self.l3(f3)
        y = F.interpolate(y, size=f2.shape[-2:], mode='bilinear', align_corners=False) + self.l2(f2)
        y = F.interpolate(y, size=f1.shape[-2:], mode='bilinear', align_corners=False) + self.l1(f1)
        y = self.cls(self.fuse(y))
        return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


def build_a3n():
    """A3 with every EfficientViT block's LiteMLA context module replaced by identity."""
    from models.compression import build_compression_student
    model = build_compression_student('A3', pretrained=False)
    n = 0
    for module in model.modules():
        if type(module).__name__ == 'EfficientViTBlock':
            module.context_module = nn.Identity(); n += 1
    assert n > 0 and not any(type(m).__name__ == 'LiteMLA' for m in model.modules())
    return model


RUNGS = {
    'R1': lambda: PerPixel(()),
    'R2': lambda: PerPixel((16, 16)),
    'R3': TinyConv,
    'R4': ContextConv,
    'A3N': build_a3n,
}


def build(graph):
    """Rungs from this file, otherwise the frozen compression ladder (A0-A3, X1, X2)."""
    if graph in RUNGS:
        return RUNGS[graph]()
    from models import compression
    compression.COMPRESSION_LADDER.update(X1=(.5, .5), X2=(.25, 1.))
    return compression.build_compression_student(graph, pretrained=False)


def per_pixel_check(model, device='cpu'):
    """Perturb one neighbour pixel; a per-pixel model must leave the centre logit unchanged."""
    model = model.to(device).eval()
    x = torch.randn(1, 6, 32, 32, device=device)
    y = x.clone(); y[..., 16, 17] += 5.
    with torch.no_grad():
        return bool(torch.equal(model(x)[..., 16, 16], model(y)[..., 16, 16]))
