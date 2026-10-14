"""Configuration for dataset loading, models, training and quantization.

Published runs use the explicit settings in experiments/, not all defaults here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.paths import dataset_root


@dataclass
class DataConfig:
    """Dataset paths and DataLoader parameters."""
    # Directory CONTAINING v1.1/, not v1.1 itself: terratorch's Sen1Floods11 datamodule
    # appends "v1.1" internally, so a root that already ends in it resolves to v1.1/v1.1.
    data_root: str = field(default_factory=lambda: str(dataset_root("S11_ROOT")))
    batch_size: int = 16
    num_workers: int = 8
    img_size: int = 224          # training crop size
    eval_img_size: int = 512     # val/test size - 512 = native resolution (official eval)
    bands: list[str] = field(default_factory=lambda: [
        "BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"
    ])


@dataclass
class ModelConfig:
    """Teacher and student architecture parameters."""
    # Teacher
    teacher_variant: str = "prithvi_eo_v2_300_tl"
    teacher_neck_indices: list[int] | None = None
    teacher_decoder_type: str = "upernet"
    teacher_decoder_channels: int = 256
    # Student
    student_arch: str = "efficientvit_b0"
    student_decoder_channels: int = 128
    # TinyViT pretrained tag (ignored for other archs)
    pretrained_tag: str = "dist_in22k_ft_in1k"
    # Shared
    num_classes: int = 2
    head_dropout: float = 0.1
    pretrained: bool = True
    teacher_ckpt: str | None = None
    official_ckpt: str | None = None   # "hf" downloads from HuggingFace, otherwise a local .pt path
    student_ckpt: str | None = None


@dataclass
class TrainConfig:
    """Training settings. DistillLoss defines the objective; experiment scripts set the paper values."""
    # Epochs & optimisation
    epochs: int = 50
    lr: float = 5e-5
    weight_decay: float = 0.05

    # Task loss
    task_loss: str = "ce"              # "ce" | "focal"
    gamma: float = 2.0                 # focal gamma (ignored when task_loss="ce")
    ignore_index: int = -1

    # Distillation hyperparameters
    alpha: float = 0.5                 # KD weight, 0..1
    temperature: float = 4.0           # KD temperature, T >= 1
    at_beta: float = 0.0               # Attention transfer weight (0 = disabled)

    # feature KD weight (used to be feat_distil_alpha). If > 0 a frozen teacher is
    # loaded and backbone features are matched with MSE.
    feat_alpha: float = 0.0

    # Concept distillation. If > 0, ConceptDistilLoss trains linear spatial probes that
    # predict NDWI/MNDWI/NDVI/EVI from neck features, and the student has to match the
    # teacher's concept encoding. 0.05 is a reasonable place to start.
    concept_distil_alpha: float = 0.0

    # feat_alpha schedule. "constant" uses feat_alpha from epoch 0, "warmup" ramps
    # linearly from 0 over feat_alpha_warmup_epochs, "decay" goes linearly down to 0
    # over feat_alpha_decay_epochs.
    feat_alpha_schedule: str = "constant"
    feat_alpha_warmup_epochs: int = 0
    feat_alpha_decay_epochs: int = 0

    # eval train, val and test after training. Mainly for pseudo-only runs: the student
    # never saw S11 train/val there, so those scores aren't inflated by memorisation.
    eval_all_splits: bool = False
    # Source-only experiments defer test/OOD readout until the finite design is frozen.
    defer_terminal_evaluation: bool = False

    # names used for per-class metrics (W&B and console), None gives class0, class1, ...
    class_names: list[str] = field(default_factory=lambda: ["background", "flood"])

    use_amp: bool = True
    amp_dtype: str = "fp16"            # "fp16" (official) | "bf16" (RTX 5090 native)

    # Gradient clipping (0 = disabled)
    grad_clip: float = 0.0

    # LR scheduler
    scheduler: str = "cosine"          # "cosine" | "step" | "none"
    step_size: int = 10
    step_gamma: float = 0.9
    warmup_epochs: int = 0

    # Early stopping on val loss (0 = disabled)
    early_stop_patience: int = 20

    val_every_n_epochs: int = 2

    # Rolling-mean window for model selection (1 = greedy best)
    val_smoothing_window: int = 3

    # Infrastructure
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    out_dir: str = "./runs"
    seed: int | None = 0
    deterministic: bool = False

    # W&B
    wandb_project: str = "EfficientSatelliteImagery"
    wandb_run_name: str | None = None



@dataclass
class QuantConfig:
    """PTQ/QAT settings. The published Jetson path uses fixed-scale, per-tensor QAT."""
    mode: str = "ptq"                  # "ptq" | "qat"
    quant_type: str = "manual_w8a8"    # "manual_w8a8" | "onnx_qdq"
    backend: str = "qnnpack"           # "qnnpack" (ARM/edge) | "fbgemm" (x86)
    calibration_batches: int = 200
    n_latency_runs: int = 50

    # QAT-specific
    qat_epochs: int = 10
    qat_lr: float = 1e-5
    weight_decay: float = 1e-5   # keep low - higher values push weights outside INT8 grid
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    val_every_n_epochs: int = 2
    use_amp: bool = False              # fake-quant + GradScaler has known issues in PyTorch 2.10
    task_loss: str = "focal"
    ignore_index: int = -1

    # TRT version compatibility for ONNX export:
    # "jp50" - JetPack 5.0 / TRT 8.4.x (Jetson Xavier): float32 weights + QDQ
    # "modern" - JetPack 5.1+ / TRT 8.5+: compact INT8 initializer form
    trt_compat: str = "jp50"

    # Output Q-DQ scale method for model_qdq_trt.onnx:
    # "calibrated" - measure max|Conv_output| from cal_batches (correct for tanh/SiLU)
    # "from_next_input" - use next layer's input scale (legacy; only safe for ReLU6)
    output_qdq_scale: str = "calibrated"

    # INT8 QDQ on the LiteMLA MatMul inputs. Default False keeps all attention MatMul in
    # FP16, same as the validated deployment model. With --matmul_qdq, ORT PTQ calibration
    # puts the mm1 inputs in INT8 and mm2 stays FP16 (its first input is the vk normalizer,
    # scale ~200-400, and quantizing that collapses attention). Only if memory is tight.
    matmul_qdq: bool = False

    # Infrastructure
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    out_dir: str = "./runs/quant"
    seed: int | None = None

    wandb_project: str = "EfficientSatelliteImagery"
    wandb_run_name: str | None = None
