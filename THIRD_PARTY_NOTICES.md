# Third-party notices

The code in this repository is under Apache-2.0 (see [LICENSE](LICENSE)). This file lists
what comes from other projects, what license applies to it, and what is only downloaded
at run time and therefore not part of this repository.

Checked in September 2026. Licenses of external projects can change; the linked pages
are the authoritative source.

## 1. Code adapted from other projects

### EfficientViT

- Source: <https://github.com/mit-han-lab/efficientvit>, commit `de7d7733cc0329f391b33f1f459271562ec27bd5`
- License: Apache-2.0
- Upstream files carry no per-file copyright header and the project ships no NOTICE file,
  so there is no header text to carry over.

Where it is used here:

| File | What was taken | Changes |
|---|---|---|
| `quantization/qat.py` (`_make_litemla_calib_hook`, `LiteMLAQAT`) | The computation of `LiteMLA.forward` and `LiteMLA.relu_linear_att` from `efficientvit/models/nn/ops.py`, rewritten step by step | Records activation ranges for calibration; both matmuls go through `QATMatMul`; the normaliser is clamped (`clamp(min=1e-6)`) instead of `+ eps` |
| `models/student.py` (`_EVIT_B_CONFIGS`, `_build_efficientvit_custom_width`) | Width, depth and `dim` values of `efficientvit_backbone_b0`/`b1` and the B0 `SegHead` settings | Widths scaled by `width_mult`; the model classes themselves are imported, not copied |

Everything else from EfficientViT is imported from the installed package.

### How this was checked

We checked TerraTorch 1.2.2,
timm 1.0.24, segmentation-models-pytorch 0.5.0, torch, torchvision and the other
installed packages), and against the reference implementations of OFA-KD
(`Hao840/OFAKD`, `distillers/ofa.py`, `_base.py`, `utils.py`), SAM (`davda54/sam`), GLISTER
(`decile-team/cords`) and the CKA demo notebook (`google-research/representation_similarity`).


The two EfficientViT adaptations above were found by reading the code, not by the scan.

Methods that are implemented here from the papers, without copying code, are cited in the
docstrings: One-for-All KD (`distillation/ofa_kd.py`), SAM (`utils/sam.py`), linear CKA
(`analysis/cka.py`) and GLISTER (`data/glister_pseudo.py`).

## 2. Libraries imported at run time (not included)

| Project | Version used | License | Link |
|---|---|---|---|
| EfficientViT | commit `de7d773` | Apache-2.0 | <https://github.com/mit-han-lab/efficientvit> |
| TerraTorch | 1.2.2 | Apache-2.0 | <https://github.com/terrastackai/terratorch> |
| timm | 1.0.24 | Apache-2.0 | <https://github.com/huggingface/pytorch-image-models> |
| segmentation-models-pytorch | 0.5.0 | MIT | <https://github.com/qubvel-org/segmentation_models.pytorch> |
| torchvision | 0.25.0 | BSD-3-Clause | <https://github.com/pytorch/vision> |

Full dependency pins are in `environment.yml`. Each package keeps its own license.

## 3. Pretrained weights (downloaded, not included)

| Weights | Used for | Source | License on the source page |
|---|---|---|---|
| Prithvi-EO-2.0-300M-TL-Sen1Floods11, file `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` | Frozen teacher | <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11> | Apache-2.0 |
| EfficientViT-B0, ImageNet-pretrained | Initialisation of the EfficientViT-B0 students | <https://huggingface.co/han-cai/efficientvit-seg> | Apache-2.0 |
| ResNet-50 ImageNet encoder (`smp-hub/resnet50.imagenet`) | UNet-ResNet50 baseline | <https://huggingface.co/smp-hub/resnet50.imagenet> | "other" (no license text on the card; original weights from `download.pytorch.org/models/resnet50-19c8e357.pth`) |
| MobileNetV3-Small `IMAGENET1K_V1` | Architecture control (Table S3) | torchvision, `download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth` | Not stated with the weights; torchvision code is BSD-3-Clause |

The files `jetson_repro/runs/deploy_qat/docker_qat/*.onnx` and `*.engine` are the trained
student of this work. They were trained starting from the ImageNet-pretrained EfficientViT-B0
checkpoint above (Apache-2.0).

## 4. Data (downloaded, not included)

| Data | Used for | Source | License / terms |
|---|---|---|---|
| STURM-Flood 1.0.0 (S2 part) | External evaluation | <https://doi.org/10.5281/zenodo.12748983> | CC BY 4.0 |
| WorldFloods-v2 (test split) | External evaluation | <https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2> | CC BY-NC 4.0 |
| Copernicus Sentinel-2 L1C (`COPERNICUS/S2_HARMONIZED`) | Raw GDACS pool | Google Earth Engine | Copernicus Sentinel Data Terms and Conditions |
| Sentinel-2 cloud probability (`COPERNICUS/S2_CLOUD_PROBABILITY`) | Patch screening | Google Earth Engine | Copernicus Sentinel Data Terms and Conditions |
| JRC Global Surface Water v1.4 (`JRC/GSW1_4/GlobalSurfaceWater`) | Permanent-water screening | Google Earth Engine | Copernicus Programme, free of charge; citation of Pekel et al. (2016) required |
| GDACS event list | Event locations and dates | <https://www.gdacs.org/> (event API used by `datasets/gee_download.py`) | GDACS disclaimer and terms of use: <https://www.gdacs.org/About/termofuse.aspx> (no separate data license stated) |

`data/ood_datasets.py` also has loaders for FloodPlanet, DeepFlood, UrbanSARFloods,
S1S2Water and SenForFlood. The paper recipes do not use them.

## 5. Images and index files

No dataset images or labels are included. The qualitative figures are rendered locally by
`jetson_repro/deploy.sh` (step `figs`) from the downloaded data; the manifests in
`jetson_repro/manifests/` only list scene file names.

`data/index_files_curations/` lists pool file names, per-tile statistics and the GDACS
location and Sentinel-2 scene ID of each tile. It holds no imagery.

`assets/pipeline_overview.png` (and `paper_source_data/Figure_01_Pipeline_Overview/pipeline.pdf`)
contains a photo of a Jetson Xavier NX board. Its source is from Fraunhofer HHI lab.
