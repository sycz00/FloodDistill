"""Main-logit adapter for the pinned author implementation of LightFormer.

The upstream checkout remains external: set LIGHTFORMER_ROOT to its directory.
Only LightFormer.py and LCRM.py are imported; upstream training/data code is unused.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from functools import lru_cache
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


SOURCE_REVISION = "574577dd639403419f511e7bab15e82a87768672"
SOURCE_HASHES = {
    "LightFormer.py": "51f02ae97926b812cfbad0f80d9831aff80fe0801ddd4bcccc25298a1813b352",
    "LCRM.py": "b297a3832feee3d773c59e810edec07f3c3f6b245e5ba69dff87b33e68d79aa5",
}


def _pad_attention_window(self, x, ps):
    """Preserve upstream reflection padding with PyTorch's required BCHW tuple."""
    height, width = x.shape[-2:]
    if width % ps:
        x = F.pad(x, (0, ps - width % ps, 0, 0), mode="reflect")
    if height % ps:
        x = F.pad(x, (0, 0, 0, ps - height % ps), mode="reflect")
    return x


@lru_cache(maxsize=None)
def _load_author_model(root: str):
    source_dir = Path(root) / "networks" / "decoder"
    for filename, expected in SOURCE_HASHES.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned LightFormer component: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"LightFormer source differs from {SOURCE_REVISION}: {path}")

    def load(filename, module_name):
        spec = importlib.util.spec_from_file_location(module_name, source_dir / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Upstream uses an absolute sibling import. Bind it only while loading the
    # model so an unrelated LCRM module elsewhere cannot silently replace it.
    lcrm = load("LCRM.py", "_flooddistill_lightformer_lcrm")
    previous = sys.modules.get("LCRM")
    sys.modules["LCRM"] = lcrm
    try:
        model_module = load("LightFormer.py", "_flooddistill_lightformer_author")
    finally:
        if previous is None:
            sys.modules.pop("LCRM", None)
        else:
            sys.modules["LCRM"] = previous
    return model_module.LightFormer


class LightFormerResNet18(nn.Module):
    """Six-band ResNet18 + published 64-channel LightFormer main decoder.

    The native eight-class training auxiliary head remains in the module for
    parameter accounting, but is unused by this project's common objectives.
    OFA heads live in OFALoss and use the four encoder outputs instead.
    """

    def __init__(self, in_channels=6, num_classes=2, head_dropout=0.1, pretrained=True):
        super().__init__()
        from models.student import _replace_first_conv

        root = os.environ.get("LIGHTFORMER_ROOT")
        if not root:
            raise RuntimeError(
                f"Set LIGHTFORMER_ROOT to the external author checkout at {SOURCE_REVISION}."
            )
        pretrained_file = os.environ.get("LIGHTFORMER_PRETRAINED_FILE") if pretrained else None
        author_model = _load_author_model(str(Path(root).resolve()))(
            backbone_name="resnet18", decode_channels=64, dropout=head_dropout,
            window_size=8, num_classes=num_classes,
            pretrained=pretrained and not pretrained_file,
        )
        self.encoder = author_model.backbone
        if pretrained_file:
            from safetensors.torch import load_file
            state = load_file(pretrained_file, device="cpu")
            result = self.encoder.load_state_dict(state, strict=False)
            if result.missing_keys or set(result.unexpected_keys) != {"fc.weight", "fc.bias"}:
                raise RuntimeError(f"Pinned ResNet18 pretrained state mismatch: {result}")
        self.decoder = author_model.decoder
        for block in (self.decoder.block1, self.decoder.block2, self.decoder.block3):
            block.GA.pad = MethodType(_pad_attention_window, block.GA)
        if in_channels != 3:
            _replace_first_conv(self.encoder, in_channels, pretrained=pretrained)
        self.encoder._ofa_stage_features_type = "spatial_list"
        self.encoder.out_channels = self.encoder.feature_info.channels()
        self.encoder.out_strides = self.encoder.feature_info.reduction()
        self.source_revision = SOURCE_REVISION
        self.source_hashes = dict(SOURCE_HASHES)
        self.native_aux_parameters = sum(p.numel() for p in self.decoder.AuxHead.parameters())
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Only this parent's flag selects the native auxiliary return path.
        # Its BatchNorm/dropout children retain mode from super().train(mode).
        self.decoder.training = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(*features, *x.shape[-2:])
