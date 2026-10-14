"""Read effective backend settings without the incompatible legacy getter.
Only diagnostic recording changes; model construction and inference are unchanged.
"""
import os
def install():
    import evaluate_v7 as ev
    def record(policy='inherited_terminal_v5'):
        import torch
        return dict(policy=policy,input_dtype='torch.float32',checkpoint_parameter_dtype='torch.float32',
          source_dtype='registered six-band normalized float32',grad_context='torch.no_grad',
          cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
          cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
          float32_matmul_precision=None,legacy_getter_omitted='PyTorch2.10 mixed-API getter raises; effective backend values recorded directly',
          effective_cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
          cudnn_conv_fp32_precision_override=torch.backends.cudnn.conv.fp32_precision,
          cudnn_fp32_precision_override=torch.backends.cudnn.fp32_precision,
          effective_cudnn_conv_fp32_precision='tf32' if torch.backends.cudnn.allow_tf32 else 'ieee',
          cudnn_benchmark=bool(torch.backends.cudnn.benchmark),cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
          deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
          backend_environment={k:os.environ.get(k) for k in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUBLAS_WORKSPACE_CONFIG')},
          torch_version=torch.__version__,pytorch_cuda_runtime=torch.version.cuda)
    ev.backend_record=record
