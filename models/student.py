"""Student segmentation models with a common (B,C,H,W) logits interface.

The paper uses EfficientViT-B0, its ReLU6 version and a UNet-ResNet50 reference;
MobileNetV3-Small and micro_m appear in the supplementary architecture control.
Other variants remain for existing callers. build_student keeps checkpoint
names and adapts the input/head for each architecture.

EfficientViT requires its source package and segmentation weights. ReLU6 is
retrained before QAT in the deployment recipe; swapping activation functions
in a trained Hardswish checkpoint is a different experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



# Shared helpers


class LeakyReLU6(nn.Module):
    """Leaky ReLU with upper clipping at 6; the negative branch remains unbounded."""

    def __init__(self, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope, inplace=False).clamp(max=6.0)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


def replace_activations(model: nn.Module, src: type, dst_factory) -> int:
    """Replace matching submodules in place; return the number replaced."""
    count = 0
    for name, mod in list(model.named_modules()):
        if type(mod) is src:
            _set_module(model, name, dst_factory())
            count += 1
    return count


def _set_module(root: nn.Module, dotted_name: str, new_mod: nn.Module) -> None:
    """Set a nested submodule by dotted name (e.g. 'backbone.stem.conv')."""
    parts  = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def _replace_first_conv(model: nn.Module, new_in_ch: int, pretrained: bool = True) -> None:
    """Replace the first Conv2d with one that takes new_in_ch input channels.

        With pretrained=True the new weights are the original RGB weights averaged over the
        channel dim and repeated for new_in_ch, so the learned spatial filters are kept.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            new_conv = nn.Conv2d(
                new_in_ch, mod.out_channels,
                kernel_size=mod.kernel_size, stride=mod.stride,
                padding=mod.padding, bias=mod.bias is not None,
            )
            if pretrained:
                # when pre-trained: copy mean values from replaced conv into the new one.
                with torch.no_grad():
                    avg = mod.weight.data.mean(dim=1, keepdim=True)   # (C_out, 1, kH, kW)
                    new_conv.weight.data = avg.expand(-1, new_in_ch, -1, -1).clone()
                    if mod.bias is not None:
                        new_conv.bias.data = mod.bias.data.clone()
            _set_module(model, name, new_conv)
            return
    raise RuntimeError("_replace_first_conv: no Conv2d found in model.")


def _replace_last_conv(model: nn.Module, num_classes: int) -> None:
    """Replace the last Conv2d (classification head) for a different num_classes."""
    last_name = last_mod = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_name, last_mod = name, mod

    if last_mod is None:
        raise RuntimeError("_replace_last_conv: no Conv2d found in model.")

    new_conv = nn.Conv2d(
        last_mod.in_channels, num_classes,
        kernel_size=last_mod.kernel_size, stride=last_mod.stride,
        padding=last_mod.padding, bias=last_mod.bias is not None,
    )
    _set_module(model, last_name, new_conv)



# MobileNetV3-Small + UNet


# feature tap nodes and their output strides for a 224x224 input
# features.1: 16ch @ H/4 (56x56)
# features.3: 24ch @ H/8 (28x28)
# features.8: 48ch @ H/16 (14x14)
# features.12: 576ch @ H/32 (7x7)
# Keep the historical extractor/skip names below for existing callers/checkpoints.
_MNV3_NODES: dict[str, str] = {
    "features.1":  "s2",
    "features.3":  "s4",
    "features.8":  "s8",
    "features.12": "s16",
}
_MNV3_CH: list[int] = [16, 24, 48, 576]


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Small with 4-scale feature extraction for 6-band input."""

    _ofa_stage_features_type = "spatial_list"

    def __init__(self, in_channels: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        from torchvision.models.feature_extraction import create_feature_extractor

        weights  = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        if in_channels != 3:
            orig = backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels, orig.out_channels,
                kernel_size=orig.kernel_size, stride=orig.stride,
                padding=orig.padding, bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    avg = orig.weight.data.mean(dim=1, keepdim=True)
                    new_conv.weight.data = avg.expand(-1, in_channels, -1, -1).clone()
            backbone.features[0][0] = new_conv

        self.extractor    = create_feature_extractor(backbone, return_nodes=_MNV3_NODES)
        self.out_channels: list[int] = _MNV3_CH
        self.out_strides: list[int] = [4, 8, 16, 32]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.extractor(x)
        return [feats["s2"], feats["s4"], feats["s8"], feats["s16"]]


class _UNetBlock(nn.Module):
    """Bilinear upsample, concat the skip, then 2x Conv-BN-ReLU6."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    """4-level UNet decoder, deepest feature to shallowest skip.

        MobileNet widths with decoder_channels=128: 256ch after up3 (H/16), 128ch
        after up2 (H/8) and 64ch after up1 (H/4), which is out_channels.
    """

    def __init__(
        self,
        enc_channels: list[int] = _MNV3_CH,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        c_s2, c_s4, c_s8, c_s16 = enc_channels
        dc = decoder_channels

        self.up3 = _UNetBlock(c_s16 + c_s8, dc * 2)
        self.up2 = _UNetBlock(dc * 2 + c_s4, dc)
        self.up1 = _UNetBlock(dc    + c_s2,  dc // 2)
        self.out_channels: int = dc // 2

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        s2, s4, s8, s16 = features
        x = self.up3(s16, s8)
        x = self.up2(x,   s4)
        x = self.up1(x,   s2)
        return x


class LightweightSegModel(nn.Module):
    """MobileNetV3-Small encoder + UNet decoder + 1x1 head.

        Attributes used by the DistillLoss AT hook: .encoder (MobileNetV3Encoder, returns
        list[Tensor]), .decoder (UNetDecoder) and .head (nn.Sequential of Dropout2d + Conv2d).
        forward(x [B, C, H, W]) returns logits [B, num_classes, H, W].
    """

    def __init__(
        self,
        encoder: MobileNetV3Encoder,
        decoder: UNetDecoder,
        num_classes: int = 2,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head    = nn.Sequential(
            nn.Dropout2d(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Conv2d(decoder.out_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W     = x.shape[-2:]
        features = self.encoder(x)          # [s2, s4, s8, s16]
        decoded  = self.decoder(features)   # (B, out_ch, H/4, W/4)
        logits   = self.head(decoded)       # (B, num_classes, H/4, W/4)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


def _build_mobilenet_seg(
    in_channels: int,
    num_classes: int,
    head_dropout: float,
    decoder_channels: int,
    pretrained: bool,
) -> LightweightSegModel:
    encoder = MobileNetV3Encoder(in_channels=in_channels, pretrained=pretrained)
    decoder = UNetDecoder(enc_channels=encoder.out_channels, decoder_channels=decoder_channels)
    return LightweightSegModel(encoder, decoder, num_classes=num_classes, head_dropout=head_dropout)



# EfficientViT


class _FullResWrapper(nn.Module):
    """EfficientViT seg model with the output upsampled to the full input resolution.

        EfficientViT seg outputs H/8. The bilinear upsample to H x W always runs; checking
        output_size == input_size first would be a data-dependent branch, and that breaks
        torch.export and FX tracing. .model is the inner EfficientViT (its .backbone has the
        stage1-4 dict) and is what the DistillLoss AT hook uses.
        forward(x [B, C, H, W]) returns logits [B, num_classes, H, W].
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        out  = self.model(x)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


def _build_efficientvit_seg(
    variant: str,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
) -> _FullResWrapper:
    """Build the pretrained segmentation backbone and return full-resolution logits.

    ReLU6 variants replace Hardswish at construction. Retrain before the paper QAT path.
    """
    try:
        from efficientvit.seg_model_zoo import create_efficientvit_seg_model as create_seg_model
    except ImportError:
        raise ImportError(
            "efficientvit is not installed.\n"
            "  pip install efficientvit\n"
            "  # or: pip install git+https://github.com/mit-han-lab/efficientvit.git"
        ) from None

    use_relu6   = variant.endswith("_relu6")
    use_silu    = variant.endswith("_silu")
    use_tanh    = variant.endswith("_tanh")
    use_lrelu6  = variant.endswith("_lrelu6")
    base_variant = (variant.removesuffix("_relu6")
                           .removesuffix("_silu")
                           .removesuffix("_tanh")
                           .removesuffix("_lrelu6"))   # "b0_lrelu6" gives "b0"

    _ADE20K_VARIANTS = {"b1", "b2", "b3", "l1", "l2"}
    dataset    = "ade20k" if base_variant in _ADE20K_VARIANTS else "cityscapes"
    model_name = f"efficientvit-seg-{base_variant}-{dataset}"
    model      = create_seg_model(model_name, pretrained=pretrained)

    if use_relu6:
        n = replace_activations(model, nn.Hardswish, lambda: nn.ReLU6(inplace=True))
        print(f"[student] efficientvit_{base_variant}_relu6: replaced {n} Hardswish → ReLU6")
    elif use_silu:
        # SiLU for deployment: no step discontinuity, so the output distribution is nearly
        # symmetric and INT8 calibration works better. Gradient >= 0.269 everywhere (never
        # dead), which fixes the proxy gradient alignment. TRT8+ fuses it as x*sigmoid(x), same
        # cost as Hardswish. With BN outputs ~N(0,1) Hardswish has a 0.13% dead zone (x<-3),
        # SiLU has none.
        n = replace_activations(model, nn.Hardswish, lambda: nn.SiLU(inplace=True))
        print(f"[student] efficientvit_{base_variant}_silu: replaced {n} Hardswish → SiLU")
    elif use_tanh:
        # Tanh is bounded to (-1, 1) in every channel, so one per-tensor INT8 scale covers all
        # channels. Smooth gradient everywhere (tanh'(x) = 1 - tanh^2(x) > 0), no dead zones for
        # the SPL-D proxy gradients, and it's a native ONNX op that TRT 8.4 supports. With BN
        # inputs ~N(0,1) the output is in (-0.96, 0.96) at +-2 sigma and the per-tensor scale is
        # ~1/127 ~ 0.008, the same for every layer and channel.
        # Tanh doesn't work with output Q-DQ INT8 deployment though: in the saturation regime
        # the output quantization step gets coarse and erases the flood-discriminative signal.
        # efficientvit_b0_lrelu6 is the deployment-compatible way to get better gradients.
        n = replace_activations(model, nn.Hardswish, lambda: nn.Tanh())
        print(f"[student] efficientvit_{base_variant}_tanh: replaced {n} Hardswish → Tanh")
    elif use_lrelu6:
        # Leaky ReLU6, max(alpha*x, x) clipped at 6. Avoids the ~50% dead neurons of ReLU6
        # (alpha=0.1 matches SiLU's negative gradient at x=-1 under BN) and keeps the max <= 6
        # bound, so the per-tensor INT8 scale is the same as for ReLU6 (~6/127 ~ 0.047). The
        # negative range is about [-0.3, 0] at +-3 sigma BN input, symmetric scale unchanged.
        # ONNX: LeakyRelu + Clip, TRT 8.4 fuses both natively.
        n = replace_activations(model, nn.Hardswish, lambda: LeakyReLU6(negative_slope=0.1))
        print(f"[student] efficientvit_{base_variant}_lrelu6: replaced {n} Hardswish → LeakyReLU6(α=0.1)")

    if in_channels != 3:
        _replace_first_conv(model, in_channels, pretrained=pretrained)

    _replace_last_conv(model, num_classes)

    return _FullResWrapper(model)


# Reference configs for the width-sweep variants, keyed by base variant (no _relu6).
# Values from efficientvit_backbone_b0/b1 and the efficientvit_seg_b0 head in EfficientViT
# (mit-han-lab/efficientvit), Apache-2.0, see THIRD_PARTY_NOTICES.md.
# width_list: channel widths per stage (stages 0-4, strides 2, 4, 8, 16, 32)
# depth_list: blocks per stage
# dim: Q/K attention dimension
_EVIT_B_CONFIGS: dict[str, dict] = {
    "b0": dict(width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2], dim=16),
    "b1": dict(width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4], dim=16),
}


def _build_efficientvit_custom_width(
    base_variant: str,
    width_mult: float,
    in_channels: int,
    num_classes: int,
) -> _FullResWrapper:
    """EfficientViT segmentation model with scaled channel widths.

        Backbone + SegHead are built from scratch without pretrained weights, there are no
        standard checkpoints for these widths. All channel widths are multiplied by
        `width_mult` (e.g. 0.25, 0.50, 0.75) and kept at >= 4 for BN stability, the attention
        dimension `dim` is scaled the same way. base_variant is "b0" or "b1" (_EVIT_B_CONFIGS).
        The first conv is adapted from 3 channels to in_channels (6 for Sentinel-2).

        Swapping activations (Hardswish to ReLU6) is left to the caller, via
        replace_activations().
    """
    try:
        from efficientvit.models.efficientvit.backbone import EfficientViTBackbone
        from efficientvit.models.efficientvit.seg import EfficientViTSeg, SegHead
    except ImportError:
        raise ImportError(
            "efficientvit is required for width-sweep variants.\n"
            "  pip install git+https://github.com/mit-han-lab/efficientvit.git"
        ) from None

    if base_variant not in _EVIT_B_CONFIGS:
        raise ValueError(
            f"Width sweep only supported for {list(_EVIT_B_CONFIGS)}, got '{base_variant}'."
        )

    cfg        = _EVIT_B_CONFIGS[base_variant]
    width_list = [max(round(w * width_mult), 4) for w in cfg["width_list"]]
    dim        = max(round(cfg["dim"] * width_mult), 4)
    depth_list = cfg["depth_list"]

    print(
        f"[student] efficientvit_{base_variant}_relu6_w{round(width_mult*100):d}: "
        f"width_list={width_list}  dim={dim}  (random init — no pretrained at this width)"
    )

    backbone = EfficientViTBackbone(
        width_list=width_list,
        depth_list=depth_list,
        in_channels=in_channels,
        dim=dim,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",   # caller swaps to relu6 via replace_activations
    )

    head_width = max(round(32 * width_mult), 8)
    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[width_list[4], width_list[3], width_list[2]],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=head_width,
        head_depth=1,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        n_classes=num_classes,
    )

    model = EfficientViTSeg(backbone, head)

    # Adapt first conv for Sentinel-2 bands (random init, no weight averaging needed)
    if in_channels != 3:
        _replace_first_conv(model, in_channels, pretrained=False)

    return _FullResWrapper(model)



# ViT-Tiny (DeiT-Tiny) segmentation student


class _ViTTinyDecoder(nn.Module):
    """Small conv decoder from ViT patch tokens to full-resolution logits.

        ViT-Tiny gives patch tokens (B, N, embed_dim) with N = (H/patch)*(W/patch), so for a
        224x224 input and patch_size=16, N = 14*14 = 196 and embed_dim = 192.

        Drops the [CLS] token, reshapes to (B, embed_dim, h_p, w_p), then 4 bilinear upsample +
        Conv-BN-GELU stages up to H x W and a 1x1 conv head for the logits. Channels with
        embed_dim=192: 192 to 96 (1/16), 96 to 64 (1/8), 64 to 32 (1/4), 32 to 32 (1/2), and
        32 to num_classes in the head.
    """

    def __init__(self, embed_dim: int = 192, num_classes: int = 2,
                 head_dropout: float = 0.1) -> None:
        super().__init__()

        def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

        self.stage0 = _up_block(embed_dim, 96)
        self.stage1 = _up_block(96, 64)
        self.stage2 = _up_block(64, 32)
        self.stage3 = _up_block(32, 32)
        self.head   = nn.Sequential(
            nn.Dropout2d(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Conv2d(32, num_classes, 1),
        )
        self.out_channels = 32

    def forward(self, tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Args:
                    tokens: (B, N+1, embed_dim), full ViT output including [CLS]
                    hw:     (H, W) of the original input image
                Returns:
                    logits (B, num_classes, H, W)
        """
        H, W   = hw
        B, NP1, D = tokens.shape
        N      = NP1 - 1                            # strip CLS
        h_p    = int(N ** 0.5)                      # 14 for 224/16
        w_p    = h_p

        x = tokens[:, 1:].permute(0, 2, 1)         # (B, D, N)
        x = x.reshape(B, D, h_p, w_p)              # (B, D, 14, 14)

        # 4 upsample stages, x2 each: 14, 28, 56, 112, 224
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)

        # Final upsample to exact (H, W) - handles non-power-of-2 inputs
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(x)


class ViTTinySegModel(nn.Module):
    """DeiT-Tiny backbone with a convolutional segmentation decoder.

    backbone is the feature-hook target; forward returns (B,num_classes,H,W) logits.
    """

    def __init__(
        self,
        backbone: nn.Module,      # timm ViT model with num_classes=0, global_pool=''
        embed_dim: int = 192,
        num_classes: int = 2,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder  = _ViTTinyDecoder(embed_dim, num_classes, head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W   = x.shape[-2:]
        tokens = self.backbone.forward_features(x)   # (B, N+1, 192)
        return self.decoder(tokens, hw=(H, W))


def _build_vit_tiny_seg(
    in_channels: int,
    num_classes: int,
    head_dropout: float,
    pretrained: bool,
) -> ViTTinySegModel:
    """ViTTinySegModel from timm's deit_tiny_patch16_224, ready for 6-channel input.

        The patch embedding is adapted from 3 to in_channels by averaging the pretrained RGB
        weights. pretrained=True downloads the ImageNet-1k weights from timm, and the channel
        average roughly keeps the pretrained spectral priors. head_dropout is applied before
        the final 1x1 conv.
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for vit_tiny.\n"
            "  pip install timm"
        ) from None

    # num_classes=0, global_pool='': forward_features() returns (B, N+1, D) tokens.
    # dynamic_img_size=True interpolates the positional embeddings at runtime, so inference
    # at 512x512 (native eval resolution) works without retraining.
    backbone = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        dynamic_img_size=True,
    )

    # patch embedding from 3 to in_channels,
    # patch_embed.proj is Conv2d(3, 192, kernel_size=16, stride=16)
    if in_channels != 3:
        old_proj   = backbone.patch_embed.proj
        new_proj   = nn.Conv2d(
            in_channels, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            bias=old_proj.bias is not None,
        )
        if pretrained:
            # Initialise by averaging RGB weights along channel dim and tiling
            with torch.no_grad():
                avg = old_proj.weight.data.mean(dim=1, keepdim=True)  # (192, 1, 16, 16)
                new_proj.weight.data = avg.expand(-1, in_channels, -1, -1).clone()
                if old_proj.bias is not None:
                    new_proj.bias.data = old_proj.bias.data.clone()
        backbone.patch_embed.proj = new_proj
        print(f"[student] vit_tiny: patch_embed adapted 3→{in_channels} channels "
              f"({'averaged pretrained weights' if pretrained else 'random init'})")

    model = ViTTinySegModel(backbone, embed_dim=192, num_classes=num_classes,
                            head_dropout=head_dropout)
    return model



# TinyViT segmentation student


# Stage output channels per TinyViT variant (strides 4, 8, 16, 32)
_TINY_VIT_CHANNELS: dict[str, list[int]] = {
    "tiny_vit_5m_224":  [ 64, 128, 160, 320],
    "tiny_vit_11m_224": [ 64, 128, 256, 448],
    "tiny_vit_21m_224": [ 96, 192, 384, 576],
}


class TinyViTSegModel(nn.Module):
    """TinyViT features_only backbone, UNet decoder and segmentation head.

    backbone returns spatial features; _stage_features_type marks the hook interface.
    """

    _stage_features_type = "spatial_list"   # consumed by backbone_distill._student_stage_features

    def __init__(
        self,
        backbone: nn.Module,          # timm features_only=True TinyViT
        enc_channels: list[int],      # [C0, C1, C2, C3]
        num_classes: int = 2,
        decoder_channels: int = 128,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder  = UNetDecoder(enc_channels=enc_channels,
                                    decoder_channels=decoder_channels)
        self.head = nn.Sequential(
            nn.Dropout2d(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W   = x.shape[-2:]
        feats  = self.backbone(x)           # list of 4 spatial maps [s4, s8, s16, s32]
        dec    = self.decoder(feats)        # (B, out_ch, H/2, W/2)
        logits = self.head(dec)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


def _build_tiny_vit_seg(
    size: str,
    in_channels: int,
    num_classes: int,
    head_dropout: float,
    decoder_channels: int,
    pretrained: bool,
    pretrained_tag: str,
) -> TinyViTSegModel:
    """Build TinyViTSegModel from timm.

        Args:
            size:           '5m', '11m', or '21m'.
            in_channels:    Input bands (6 for Sentinel-2). First Conv2d adapted from 3 to N.
            num_classes:    Segmentation output classes.
            head_dropout:   Dropout before final 1x1 conv.
            decoder_channels: UNet decoder width (default 128).
            pretrained:     If False, random init.
            pretrained_tag: timm pretrained tag. Best options:
                            'dist_in22k_ft_in1k' (default, highest accuracy)
                            'dist_in22k' (most general features, no task fine-tune)
                            'in1k' (ImageNet-1k only)
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for tiny_vit.\n  pip install timm") from None

    timm_name = f"tiny_vit_{size}_224"
    if timm_name not in _TINY_VIT_CHANNELS:
        raise ValueError(f"Unknown TinyViT size '{size}'. Choose from: 5m, 11m, 21m")

    model_name = f"{timm_name}.{pretrained_tag}" if pretrained else timm_name
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
    )

    enc_channels = _TINY_VIT_CHANNELS[timm_name]

    # first Conv2d in patch_embed from 3 to in_channels,
    # patch_embed.conv1.conv is Conv2d(3, 32/48, 3x3, stride=2)
    if in_channels != 3:
        old_conv = backbone.patch_embed.conv1.conv
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                avg = old_conv.weight.data.mean(dim=1, keepdim=True)
                new_conv.weight.data = avg.expand(-1, in_channels, -1, -1).clone()
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data.clone()
        backbone.patch_embed.conv1.conv = new_conv
        print(f"[student] tiny_vit_{size}: patch_embed adapted 3→{in_channels} channels "
              f"({'averaged pretrained weights' if pretrained else 'random init'})")

    return TinyViTSegModel(
        backbone=backbone,
        enc_channels=enc_channels,
        num_classes=num_classes,
        decoder_channels=decoder_channels,
        head_dropout=head_dropout,
    )



# Generic timm features_only wrapper


class TimmFeaturesSegModel(nn.Module):
    """timm features_only backbone + UNetDecoder + head.

        Works with any timm model that gives >= 4 spatial feature maps with features_only=True.
        Registered here: lcnet_050, lcnet_075, lcnet_050_relu6, mobilevit_xxs.

        For the hooks: .backbone is the timm features_only model (returns list[Tensor]), and
        _stage_features_type = "spatial_list" like TinyViTSegModel, which selects the
        spatial-list hook path in losses.py. forward(x [B, C, H, W]) returns logits
        [B, num_classes, H, W].
    """

    _stage_features_type = "spatial_list"   # hook path in training/losses.py

    def __init__(
        self,
        backbone: nn.Module,
        decoder: "UNetDecoder",
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder  = decoder
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W   = x.shape[-2:]
        feats  = self.backbone(x)           # list[Tensor[B, C, H, W]], 4 of them
        dec    = self.decoder(feats)        # (B, out_ch, H/4, W/4)
        logits = self.head(dec)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


def _build_timm_seg(
    arch: str,
    in_channels: int,
    num_classes: int,
    head_dropout: float,
    decoder_channels: int,
    pretrained: bool,
) -> TimmFeaturesSegModel:
    """Segmentation model from any timm features_only backbone (e.g. "lcnet_050", "mobilevit_xxs").

        Uses feature_info to pick the last 4 feature levels (strides 4, 8, 16, 32), adapts the
        first Conv2d to in_channels by weight averaging and adds UNetDecoder (decoder_channels,
        default 128) + 1x1 head. pretrained loads ImageNet weights.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required.\n  pip install timm") from None

    # count stages without a weight download; fast, no GPU needed
    _probe = timm.create_model(arch, pretrained=False, features_only=True)
    n_feats = len(_probe.feature_info)
    if n_feats < 4:
        raise ValueError(
            f"_build_timm_seg: '{arch}' has only {n_feats} feature levels; need ≥4."
        )
    out_indices = tuple(range(n_feats - 4, n_feats))   # last 4 stages, strides 4 to 32
    del _probe

    # Build with pretrained weights and selected output indices
    backbone = timm.create_model(
        arch,
        pretrained=pretrained,
        features_only=True,
        out_indices=out_indices,
    )

    # Adapt first Conv2d: average RGB pretrained weights to new_in_ch channels
    if in_channels != 3:
        _replace_first_conv(backbone, in_channels, pretrained=pretrained)

    channels = list(backbone.feature_info.channels())   # [C_s4, C_s8, C_s16, C_s32]
    decoder  = UNetDecoder(enc_channels=channels, decoder_channels=decoder_channels)
    head     = nn.Sequential(
        nn.Dropout2d(head_dropout) if head_dropout > 0 else nn.Identity(),
        nn.Conv2d(decoder.out_channels, num_classes, kernel_size=1),
    )
    return TimmFeaturesSegModel(backbone, decoder, head)



# Micro-architectures (pure Q-easy blocks, DLA-compatible)


def _build_micro_seg(
    size: str,
    in_channels: int,
    num_classes: int,
    head_dropout: float,
) -> TimmFeaturesSegModel:
    """Micro segmentation model from pure Q-easy (DLA-compatible) blocks.

        MicroBackbone from models/qblocks.py, decoder channels and stage configs from
        MICRO_CONFIGS, wrapped in TimmFeaturesSegModel so forward() is the same.
        size is "s" (small, DWSep) or "m" (medium, MBConv).
    """
    from models.qblocks import MicroBackbone, MICRO_CONFIGS

    key = f"micro_{size}"
    if key not in MICRO_CONFIGS:
        raise ValueError(f"Unknown micro size '{size}'. Choose from: s, m")

    config   = MICRO_CONFIGS[key]
    backbone = MicroBackbone(in_channels=in_channels, config=config)
    decoder  = UNetDecoder(
        enc_channels=backbone.out_channels,
        decoder_channels=config.get("decoder_channels", 32),
    )
    head = nn.Sequential(
        nn.Dropout2d(head_dropout) if head_dropout > 0 else nn.Identity(),
        nn.Conv2d(decoder.out_channels, num_classes, kernel_size=1),
    )
    # MicroBackbone already has _stage_features_type = "spatial_list";
    # TimmFeaturesSegModel exposes it as .backbone for hook registration.
    return TimmFeaturesSegModel(backbone, decoder, head)



# U-Net + ResNet baselines


_BASELINE_PARAMS: dict[str, str] = {
    "unet_resnet50":  "resnet50",
    "unet_resnet101": "resnet101",
}


def _build_unet_resnet(
    arch: str,
    in_channels: int,
    num_classes: int,
) -> nn.Module:
    """U-Net with a ResNet-50 or ResNet-101 encoder (segmentation-models-pytorch), ImageNet weights.

        For in_channels != 3 the first Conv2d gets the pretrained RGB weights averaged over the
        channel dim, same as _replace_first_conv. Returns raw logits (activation=None).
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch is required for U-Net baselines.\n"
            "  pip install segmentation-models-pytorch"
        ) from None

    if arch not in _BASELINE_PARAMS:
        raise ValueError(f"Unknown baseline arch '{arch}'. Choices: {list(_BASELINE_PARAMS)}")

    encoder_name = _BASELINE_PARAMS[arch]
    print(f"[student] building U-Net + {encoder_name}")

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,    # raw logits - consistent with all other models
    )



# Unified factory


def build_student(
    arch: str,
    in_channels: int = 6,
    num_classes: int = 2,
    head_dropout: float = 0.1,
    decoder_channels: int = 128,
    pretrained: bool = True,
    pretrained_tag: str = "dist_in22k_ft_in1k",
    ckpt_path: str | None = None,
) -> nn.Module:
    """Build a student and optionally load ckpt_path after architecture adaptation.

    All variants return logits=(B,num_classes,H,W). pretrained selects the
    architecture's upstream weights; pretrained_tag applies to TinyViT only.
    decoder_channels controls compatible UNet decoders; micro uses its own config.
    Use the exact architecture and activation of the saved checkpoint.
    """
    from models.qblocks import print_quant_summary

    arch_lower = arch.lower()

    # Micro-architectures (pure Q-easy, DLA-compatible)
    if arch_lower in ("micro_s", "micro_m"):
        size  = arch_lower[len("micro_"):]   # "s" or "m"
        model = _build_micro_seg(
            size=size,
            in_channels=in_channels,
            num_classes=num_classes,
            head_dropout=head_dropout,
        )

    # timm LCNet variants
    elif arch_lower in ("lcnet_050", "lcnet_075", "lcnet_050_relu6"):
        base  = arch_lower.replace("_relu6", "")   # "lcnet_050_relu6" gives "lcnet_050"
        model = _build_timm_seg(
            arch=base,
            in_channels=in_channels,
            num_classes=num_classes,
            head_dropout=head_dropout,
            decoder_channels=decoder_channels,
            pretrained=pretrained,
        )
        if arch_lower.endswith("_relu6"):
            n  = replace_activations(model, nn.Hardswish,   lambda: nn.ReLU6(inplace=True))
            n += replace_activations(model, nn.SiLU,        lambda: nn.ReLU6(inplace=True))
            n += replace_activations(model, nn.Hardsigmoid, lambda: nn.ReLU6(inplace=True))
            if n > 0:
                print(f"[student] {arch_lower}: replaced {n} medium activations → ReLU6 (DLA-ready)")

    # timm MobileViT-XXS
    elif arch_lower == "mobilevit_xxs":
        model = _build_timm_seg(
            arch=arch_lower,
            in_channels=in_channels,
            num_classes=num_classes,
            head_dropout=head_dropout,
            decoder_channels=decoder_channels,
            pretrained=pretrained,
        )

    # MobileNetV3 (torchvision)
    elif arch_lower == "mobilenet_v3_small":
        model = _build_mobilenet_seg(
            in_channels=in_channels,
            num_classes=num_classes,
            head_dropout=head_dropout,
            decoder_channels=decoder_channels,
            pretrained=pretrained,
        )

    elif arch_lower == "lightformer_resnet18":
        from models.lightformer import LightFormerResNet18
        model = LightFormerResNet18(
            in_channels=in_channels, num_classes=num_classes,
            head_dropout=head_dropout, pretrained=pretrained,
        )

    # EfficientViT
    elif arch_lower.startswith("efficientvit_"):
        import re
        variant = arch_lower[len("efficientvit_"):]   # "b0", "b1", "b0_relu6", "b0_relu6_w50"

        # width-sweep variant, e.g. "b0_relu6_w50" means width_mult=0.50, relu6=True
        wm_match = re.match(r"^(.+?)_w(\d+)$", variant)
        if wm_match:
            base_with_flags = wm_match.group(1)                    # "b0_relu6" or "b0"
            width_mult       = int(wm_match.group(2)) / 100.0      # 0.50
            use_relu6        = base_with_flags.endswith("_relu6")
            base_variant     = base_with_flags.removesuffix("_relu6")   # "b0"
            model = _build_efficientvit_custom_width(
                base_variant=base_variant,
                width_mult=width_mult,
                in_channels=in_channels,
                num_classes=num_classes,
            )
            if use_relu6:
                n  = replace_activations(model, nn.Hardswish,   lambda: nn.ReLU6(inplace=True))
                n += replace_activations(model, nn.Hardsigmoid, lambda: nn.ReLU6(inplace=True))
                if n > 0:
                    print(f"[student] {arch_lower}: replaced {n} activations → ReLU6 (DLA-ready)")
        else:
            model = _build_efficientvit_seg(
                variant=variant,
                in_channels=in_channels,
                num_classes=num_classes,
                pretrained=pretrained,
            )

    # ViT-Tiny (DeiT)
    elif arch_lower == "vit_tiny":
        model = _build_vit_tiny_seg(
            in_channels=in_channels,
            num_classes=num_classes,
            head_dropout=head_dropout,
            pretrained=pretrained,
        )

    # TinyViT (hybrid CNN+ViT)
    elif arch_lower.startswith("tiny_vit_"):
        size  = arch_lower[len("tiny_vit_"):]   # "5m", "11m", "21m"
        model = _build_tiny_vit_seg(
            size=size,
            in_channels=in_channels,
            num_classes=num_classes,
            head_dropout=head_dropout,
            decoder_channels=decoder_channels,
            pretrained=pretrained,
            pretrained_tag=pretrained_tag,
        )
        if pretrained:
            print(f"[student] tiny_vit_{size}: pretrained_tag={pretrained_tag}")

    # U-Net + ResNet baselines
    elif arch_lower.startswith("unet_"):
        model = _build_unet_resnet(
            arch=arch_lower,
            in_channels=in_channels,
            num_classes=num_classes,
        )

    else:
        raise ValueError(
            f"Unknown student arch '{arch}'.\n"
            "Q-easy/DLA: 'micro_s', 'micro_m', 'lcnet_050_relu6'\n"
            "Q-medium:   'lcnet_050', 'lcnet_075', 'mobilevit_xxs',\n"
            "            'mobilenet_v3_small', 'efficientvit_b0[_relu6]',\n"
            "            'efficientvit_b1', 'efficientvit_b2'\n"
            "Q-hard:     'vit_tiny', 'tiny_vit_5m', 'tiny_vit_11m', 'tiny_vit_21m'\n"
            "Baselines:  'unet_resnet50', 'unet_resnet101', 'lightformer_resnet18'"
        )

    # Print param count + quantization tier summary
    print_quant_summary(arch_lower, model)

    if ckpt_path:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state   = payload.get("student", payload.get("model", payload))
        model.load_state_dict(state, strict=True)
        print(f"[student] loaded checkpoint: {ckpt_path}")

    return model
