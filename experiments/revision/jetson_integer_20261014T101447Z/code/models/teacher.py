"""Prithvi teacher: backbone + neck + decoder + head in one nn.Module.

TeacherModel exposes named nn.Module submodules for the AT hooks: .backbone, .neck,
.decoder, .head. build_teacher creates it, with optional loading and freezing. load_official_ckpt
loads the ibm-nasa-geospatial HF Lightning checkpoint into a TeacherModel.

Shapes:
    input                              (B, C, H, W)
    backbone.forward_features()        list[Tensor(B, 1+N, D)]   N patches, D embed_dim
    neck (SelectIndices + Reshape)     list[4 x Tensor(B, D, H/p, W/p)]
    decoder (UPerNet + scale_modules)  Tensor(B, dec_ch, H/4, W/4)
    head (dropout + 1x1 conv)          Tensor(B, num_classes, H/4, W/4)
    bilinear upsample                  Tensor(B, num_classes, H, W)

scale_modules=True in UperNetDecoder builds the fpn1/fpn2 ConvTranspose2d layers that turn
the flat 14x14 ViT patch tokens into a proper multi-scale FPN pyramid, as in the official
ibm-nasa-geospatial checkpoint.

Key mapping for the official HF checkpoint (load_official_ckpt):
    model.encoder.X  to  backbone.X
    model.decoder.X  to  decoder.X
    model.head.X     to  head.X
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import (
    prithvi_eo_v2_300,
    prithvi_eo_v2_300_tl,
    prithvi_eo_v2_600,
    prithvi_eo_v2_600_tl,
)
from terratorch.models.necks import NeckSequential, ReshapeTokensToImage, SelectIndices
from terratorch.models.decoders.fcn_decoder import FCNDecoder
from terratorch.models.decoders.upernet_decoder import UperNetDecoder
from terratorch.models.heads.segmentation_head import SegmentationHead



# Constants


BAND_NAME_MAP: dict[str, HLSBands] = {
    "BLUE":        HLSBands.BLUE,
    "GREEN":       HLSBands.GREEN,
    "RED":         HLSBands.RED,
    "NIR_NARROW":  HLSBands.NIR_NARROW,
    "SWIR_1":      HLSBands.SWIR_1,
    "SWIR_2":      HLSBands.SWIR_2,
    "NIR_BROAD":   HLSBands.NIR_BROAD,
    "RED_EDGE_1":  HLSBands.RED_EDGE_1,
    "RED_EDGE_2":  HLSBands.RED_EDGE_2,
    "RED_EDGE_3":  HLSBands.RED_EDGE_3,
}

_FACTORY_MAP = {
    "prithvi_eo_v2_300":    prithvi_eo_v2_300,
    "prithvi_eo_v2_300_tl": prithvi_eo_v2_300_tl,
    "prithvi_eo_v2_600":    prithvi_eo_v2_600,
    "prithvi_eo_v2_600_tl": prithvi_eo_v2_600_tl,
}

_NECK_INDICES: dict[str, list[int]] = {
    "prithvi_eo_v2_300":    [5, 11, 17, 23],
    "prithvi_eo_v2_300_tl": [5, 11, 17, 23],
    "prithvi_eo_v2_600":    [7, 15, 23, 31],
    "prithvi_eo_v2_600_tl": [7, 15, 23, 31],
}

# Official IBM/NASA fine-tuned checkpoints on HuggingFace. Only variants with a released full
# segmentation checkpoint (backbone+decoder+head) are listed, load_official_ckpt(variant=...)
# downloads them. 600M-TL has no Sen1Floods11 checkpoint yet, fine-tune it yourself and pass
# the local path with --teacher_ckpt.
_OFFICIAL_HF_CKPTS: dict[str, tuple[str, str]] = {
    # variant: (hf_repo, filename)
    "prithvi_eo_v2_300_tl": (
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
    ),
    # Add entries here when IBM/NASA release 600M fine-tuned checkpoints.
}



# TeacherModel


class TeacherModel(nn.Module):
    """Backbone + neck + decoder + head in one nn.Module.

        The submodule names stay fixed because DistillLoss registers AT hooks by attribute name:
            .backbone   PrithviViT feature extractor
            .neck       ViTNeck (SelectIndices + ReshapeTokensToImage)
            .decoder    UperNetDecoder (scale_modules=True for the multi-scale FPN)
            .head       SegmentationHead

        forward(x) returns logits [B, num_classes, H, W].
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck     = neck
        self.decoder  = decoder
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        features  = self.backbone.forward_features(x)   # list[Tensor(B, 1+N, D)]
        neck_out  = self.neck(features)                  # list[4 x Tensor(B, D, H/p, W/p)]
        decoded   = self.decoder(neck_out)               # Tensor(B, dec_ch, H/4, W/4)
        logits    = self.head(decoded)                   # Tensor(B, num_classes, H/4, W/4)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)



# ViTNeck


class _ViTNeck(nn.Module):
    """SelectIndices + ReshapeTokensToImage in one module.

        Shapes for T=1, H x W input and the 300M backbone:
            input                         list[Tensor(B, 1+N, 1024)]  (24 blocks)
            SelectIndices([5,11,17,23])   list[4 x Tensor(B, 1+N, 1024)]
            ReshapeTokensToImage          list[4 x Tensor(B, 1024, H/16, W/16)]
    """

    def __init__(
        self,
        backbone_out_channels: list[int],
        neck_indices: list[int],
        effective_time_dim: int = 1,
    ) -> None:
        super().__init__()
        select  = SelectIndices(backbone_out_channels, neck_indices)
        ch_post = select.process_channel_list(backbone_out_channels)
        reshape = ReshapeTokensToImage(
            ch_post,
            remove_cls_token=True,
            effective_time_dim=effective_time_dim,
        )
        self.seq = NeckSequential(select, reshape)
        self.out_channels: list[int] = reshape.process_channel_list(ch_post)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.seq(features)



# Official checkpoint loader


def load_official_ckpt(
    model: TeacherModel,
    ckpt_path: str | None = None,
    variant: str = "prithvi_eo_v2_300_tl",
) -> TeacherModel:
    """Load an ibm-nasa-geospatial HF fine-tuned checkpoint into a TeacherModel.

        The official checkpoint is a PyTorch Lightning state_dict. Its keys model.encoder.X,
        model.decoder.X and model.head.X become backbone.X, decoder.X and head.X.

        `model` should come from build_teacher(pretrained=False). Without ckpt_path the file is
        downloaded from HuggingFace, looked up by `variant`; variants without a released
        segmentation checkpoint (e.g. prithvi_eo_v2_600_tl) raise ValueError.
        Returns the same model instance with the official weights loaded (strict=True).
    """
    if ckpt_path is None:
        if variant not in _OFFICIAL_HF_CKPTS:
            available = list(_OFFICIAL_HF_CKPTS.keys())
            raise ValueError(
                f"No official Sen1Floods11 checkpoint for variant '{variant}'. "
                f"Available: {available}. "
                f"Fine-tune the model yourself and pass --teacher_ckpt <path>."
            )
        from huggingface_hub import hf_hub_download
        hf_repo, hf_file = _OFFICIAL_HF_CKPTS[variant]
        ckpt_path = hf_hub_download(hf_repo, hf_file)

    raw = torch.load(ckpt_path, map_location="cpu")

    # Handle our own saved format: {"model": already-remapped state_dict, ...}
    if "model" in raw and isinstance(raw["model"], dict):
        sample = next(iter(raw["model"]), "")
        if sample.startswith(("backbone.", "decoder.", "head.")):
            missing, unexpected = model.load_state_dict(raw["model"], strict=True)
            print(f"Loaded local checkpoint: {len(raw['model'])} keys, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")
            return model

    src_sd = raw["state_dict"] if "state_dict" in raw else raw

    # Strip the "model." prefix and rename "encoder." to "backbone.".
    # The official checkpoint keeps the UPerNet scale modules (fpn1/fpn2) in the neck as
    # "model.neck.2.fpn*". terratorch >= 1.2 builds them there as well, but we pass
    # scale_modules=True to UperNetDecoder, so here they sit under decoder.*. Same tensors and
    # shapes, only the owner is different.
    dst_sd: dict[str, torch.Tensor] = {}
    for k, v in src_sd.items():
        if not k.startswith("model."):
            continue
        k = k[len("model."):]           # strip "model."
        if k.startswith("encoder."):
            k = "backbone." + k[len("encoder."):]
        elif k.startswith("neck.2.fpn"):
            k = "decoder." + k[len("neck.2."):]
        dst_sd[k] = v

    missing, unexpected = model.load_state_dict(dst_sd, strict=True)
    # strict=True raises on any mismatch, catching architecture bugs early
    print(f"Loaded official checkpoint: {len(dst_sd)} keys, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    return model



# build_teacher factory


def build_teacher(
    variant: str = "prithvi_eo_v2_300_tl",
    bands: list[str] | None = None,
    neck_indices: list[int] | None = None,
    decoder_type: str = "upernet",
    decoder_channels: int = 256,
    num_classes: int = 2,
    head_dropout: float = 0.1,
    pretrained: bool = True,
    ckpt_path: str | None = None,
    freeze: bool = False,
    official_ckpt: str | None = None,
) -> TeacherModel:
    """Build a TeacherModel (Prithvi backbone + neck + decoder + head).

        Args:
            variant:          Prithvi backbone variant name.
            bands:            Band name strings; defaults to 6 standard S2 bands.
            neck_indices:     Transformer block indices to tap; defaults to variant-standard.
            decoder_type:     "upernet" (default) or "fcn".
            decoder_channels: UperNet output channels (256 matches official config).
            num_classes:      Segmentation output classes.
            head_dropout:     Dropout in SegmentationHead.
            pretrained:       Load pretrained backbone weights from HuggingFace.
            ckpt_path:        Path to backbone .pt for fine-tuned backbone weights.
            freeze:           Freeze all parameters (use as inference-only teacher).
            official_ckpt:    Path to official Lightning checkpoint; if set, calls
                              load_official_ckpt() after building (pretrained is ignored).

        Returns:
            TeacherModel with .backbone, .neck, .decoder, .head attributes.
    """
    if variant not in _FACTORY_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_FACTORY_MAP.keys())}")

    if bands is None:
        bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

    if neck_indices is None:
        neck_indices = _NECK_INDICES.get(variant, [5, 11, 17, 23])

    # Backbone
    hls_bands   = [BAND_NAME_MAP[b] for b in bands]
    factory_fn  = _FACTORY_MAP[variant]
    bb_kwargs: dict = {"num_frames": 1}
    if ckpt_path is not None:
        bb_kwargs["ckpt_path"] = ckpt_path

    load_pretrained = pretrained and (official_ckpt is None)
    backbone = factory_fn(pretrained=load_pretrained, bands=hls_bands, **bb_kwargs)

    # Neck
    neck = _ViTNeck(backbone.out_channels, neck_indices)

    # Decoder
    # scale_modules=True builds fpn1/fpn2 ConvTranspose2d inside UperNetDecoder,
    # matching the official ibm-nasa-geospatial checkpoint architecture exactly.
    if decoder_type == "upernet":
        decoder = UperNetDecoder(
            embed_dim=neck.out_channels,
            channels=decoder_channels,
            pool_scales=(1, 2, 3, 6),
            scale_modules=True,
        )
    elif decoder_type == "fcn":
        decoder = FCNDecoder(
            embed_dim=neck.out_channels[-1],
            channels=decoder_channels,
            num_convs=4,
            in_index=-1,
        )
    else:
        raise ValueError(f"Unknown decoder_type '{decoder_type}'. Choose 'upernet' or 'fcn'.")

    # Head
    head = SegmentationHead(
        in_channels=decoder.out_channels,
        num_classes=num_classes,
        channel_list=None,
        dropout=head_dropout,
    )

    model = TeacherModel(backbone, neck, decoder, head)

    # Load official checkpoint (overrides backbone pretrained weights)
    if official_ckpt is not None:
        load_official_ckpt(
            model,
            ckpt_path=official_ckpt if official_ckpt != "hf" else None,
            variant=variant,
        )

    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    return model
