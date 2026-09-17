"""Channel adapters for evaluating the 6-band optical Sentinel-2 models on other datasets.

Dataset               Sensor  Bands  Direct?  Adapter mode
STURM-Flood  (S2)     S2      9      yes*     band_select [0,1,2,6,7,8]
STURM-Flood  (S1)     SAR     2      no       pad_zeros or learned
DeepFlood    (SAR)    SAR     2      no       pad_zeros or learned
UrbanSARFloods        SAR     8      no       band_select or learned

* after picking the 6 bands that match the training data. Accuracy will still differ
  from Sen1Floods11 because of domain shift.

pad_zeros and band_select have no learnable parameters and work for zero-shot evaluation.
learned has to be trained on labeled data first, so it's not for zero-shot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ChannelAdapter(nn.Module):
    """Map N input channels to target_channels, spatial size unchanged.

    band_select picks band_indices (length must be target_channels), no params, zero-shot ok.
    pad_zeros takes the first min(in_ch, target) bands and fills the rest with 0.
    learned is a 1x1 conv, has to be trained before use.
    """

    def __init__(
        self,
        in_channels: int,
        target_channels: int = 6,
        mode: str = "band_select",
        band_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.target_channels = target_channels
        self.mode = mode

        if mode == "band_select":
            if band_indices is None:
                raise ValueError("band_indices required for 'band_select' mode")
            if len(band_indices) != target_channels:
                raise ValueError(
                    f"band_indices has {len(band_indices)} entries "
                    f"but target_channels={target_channels}"
                )
            self.register_buffer("band_indices", torch.tensor(band_indices, dtype=torch.long))

        elif mode == "pad_zeros":
            pass  # stateless

        elif mode == "learned":
            self.proj = nn.Conv2d(in_channels, target_channels, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.proj.weight)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'band_select', 'pad_zeros', or 'learned'.")

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, H, W) -> (B, target_channels, H, W)"""
        if self.mode == "band_select":
            return x[:, self.band_indices]

        if self.mode == "pad_zeros":
            B, C, H, W = x.shape
            take = min(C, self.target_channels)
            out = torch.zeros(B, self.target_channels, H, W, dtype=x.dtype, device=x.device)
            out[:, :take] = x[:, :take]
            return out

        # learned
        return self.proj(x)


def adapt_batch(batch: dict, adapter: ChannelAdapter) -> dict:
    """Apply ChannelAdapter to batch["image"] in-place (returns new dict)."""
    return {**batch, "image": adapter(batch["image"])}


# ready-made adapters

def sturm_s2_adapter(target_channels: int = 6) -> ChannelAdapter:
    """Pick the 6 Sen1Floods11 training bands out of the 9 STURM-Flood S2 bands.

    Assumed STURM order: 0=B2 (blue), 1=B3 (green), 2=B4 (red), 3=B5, 4=B6, 5=B7,
    6=B8A (NIR narrow), 7=B11 (SWIR1), 8=B12 (SWIR2). BLUE, GREEN, RED, NIR_NARROW,
    SWIR_1, SWIR_2 are then indices [0, 1, 2, 6, 7, 8].

    Double-check against the .tif metadata, rasterio.open(path).descriptions
    gives the band names.
    """
    return ChannelAdapter(
        in_channels=9,
        target_channels=target_channels,
        mode="band_select",
        band_indices=[0, 1, 2, 6, 7, 8],
    )


def sar_2band_adapter(target_channels: int = 6) -> ChannelAdapter:
    """Zero-pad 2-band SAR (VV, VH) to target_channels.

    Only for zero-shot domain-gap checks. The model never saw SAR, expect bad numbers.
    """
    return ChannelAdapter(in_channels=2, target_channels=target_channels, mode="pad_zeros")


def urban_sar_adapter(target_channels: int = 6) -> ChannelAdapter:
    """Co-event VV/VH intensity from UrbanSARFloods (8 bands), the most
    flood-informative pair.

    Band order: 0=VV_pre_intensity, 1=VH_pre_intensity, 2=VV_co_intensity,
    3=VH_co_intensity, 4=VV_pre_coherence, 5=VH_pre_coherence,
    6=VV_post_intensity, 7=VH_post_intensity
    """
    return ChannelAdapter(
        in_channels=8,
        target_channels=target_channels,
        mode="band_select",
        band_indices=[2, 3, 2, 3, 2, 3],  # co-event VV/VH repeated 3x for the 6 slots
    )


def print_compatibility_report() -> None:
    """Print a human-readable compatibility table for all supported datasets."""
    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│  Dataset Compatibility with 6-channel Sentinel-2 optical models             │
├──────────────────────┬────────┬───────┬───────────────────────────────────  │
│  Dataset             │ Sensor │ Bands │ Adapter / Notes                      │
├──────────────────────┼────────┼───────┼──────────────────────────────────── │
│  STURM-Flood  (S2)   │  S2    │   9   │ sturm_s2_adapter() — band select    │
│                      │        │       │ Same sensor as training → best OOD  │
├──────────────────────┼────────┼───────┼──────────────────────────────────── │
│  STURM-Flood  (S1)   │  SAR   │   2   │ sar_2band_adapter() — pad zeros     │
│                      │        │       │ Different sensor — expect poor perf  │
├──────────────────────┼────────┼───────┼──────────────────────────────────── │
│  DeepFlood (SAR)     │  SAR   │   2   │ sar_2band_adapter() — pad zeros     │
│                      │        │       │ Different sensor — expect poor perf  │
├──────────────────────┼────────┼───────┼──────────────────────────────────── │
│  UrbanSARFloods      │  SAR   │   8   │ urban_sar_adapter() — band select   │
│                      │        │       │ Different sensor — expect poor perf  │
└──────────────────────┴────────┴───────┴──────────────────────────────────── ┘

To verify actual band names in a .tif file:
    import rasterio
    with rasterio.open(path) as src:
        print(src.descriptions)   # band names if embedded
        print(src.meta)           # dtype, count, crs, etc.
"""
    )
