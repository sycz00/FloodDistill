"""Build a QDQ-annotated ONNX model from a float model, using the per-channel INT8
scales calibrated by the manual_w8a8 pipeline (manual.py).

Q-DQ per Conv node:
  data input  QuantizeLinear(per-ch, axis=1), DequantizeLinear(per-ch, axis=1), then Conv
  weight      [int8 initializer], DequantizeLinear(per-ch, axis=0), then Conv
  output      unchanged float

No output Q-DQ on purpose, so the numbers behave like manual_w8a8. onnxruntime on x86 runs
a float32 Conv after the DQL (same ~0.794 mIoU), TensorRT on the Jetson fuses the Q-DQ
hints into INT8/FP16 Tensor Core kernels.

Usage
    from quantization.onnx_qdq import build_qdq_onnx, eval_onnx

    float_path, qdq_path = build_qdq_onnx(
        model, cal_batches, out_dir="runs/deploy"
    )
    metrics = eval_onnx(qdq_path, val_batches, num_classes=2)

Standalone export, if you already have calibrated quant_convs:
    float_path = export_float_onnx(model, dummy_input, "out/float.onnx")
    qdq_path   = inject_qdq(float_path, quant_convs, "out/qdq.onnx")
"""

from __future__ import annotations

import copy
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from quantization.manual import (
    QuantConv2d, fold_bn, quantize_model,
    calibrate_activations, calibrate_output_scales, apply_smooth_quant,
)



# ONNX simplification helper (shared with qat.py)


def _simplify_onnx(src: str, dst: str, input_shapes: dict | None = None) -> bool:
    """Run onnx-simplifier from src to dst, returns True on success.

        Needed for jp50/TRT 8.4: LiteMLA attention calls qkv.size() to get H*W for a reshape,
        which ends up as a Shape->Gather->Mul chain in ONNX, and TRT 8.4 rejects IShuffleLayer
        on shape tensors with >1 dim. onnxsim constant-folds H*W (fixed at export time), so all
        those shape-tensor ops go away. Also switches bicubic Resize to bilinear (TRT 8.4 has no
        cubic upsample).
    """
    try:
        import onnxsim
        import onnx as _onnx
    except ImportError:
        print("[onnx_qdq] WARNING: onnx-simplifier not installed -- "
              "model_qdq_trt.onnx may fail in TRT 8.4 due to LiteMLA shape ops.\n"
              "  Fix: pip install onnx-simplifier")
        return False

    model = _onnx.load(src)
    kwargs = {}
    if input_shapes:
        kwargs["input_shapes"] = input_shapes
    simplified, ok = onnxsim.simplify(model, **kwargs)
    if not ok:
        print("[onnx_qdq] WARNING: onnxsim reported simplification failed; "
              "TRT QDQ will be built from unsimplified base (may crash TRT 8.4)")
        return False

    # TRT 8.4 does not support bicubic interpolation.
    cubic_count = 0
    for node in simplified.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "mode" and attr.s == b"cubic":
                attr.s = b"linear"
                cubic_count += 1
    if cubic_count:
        print(f"  onnxsim: patched {cubic_count} Resize node(s) cubic->linear "
              "(TRT 8.4 does not support bicubic)")

    _onnx.save(simplified, dst)
    before = os.path.getsize(src) // 1024
    after  = os.path.getsize(dst) // 1024
    print(f"  onnxsim: {os.path.basename(src)} ({before} KB) -> "
          f"{os.path.basename(dst)} ({after} KB)  [shape-tensor ops folded]")
    return True


def _write_fp16_mm2_sidecar(src: str, dst: str) -> None:
    """Write the LiteMLA mm2 normalization node names (one per line) to `dst`.

        benchmark_jetson.py reads this to pin those nodes to FP32 in the FP16 engine.
        Loads jetson/fp16_mm2_layers.py by path (jetson/ is not a package). Never
        raises: any failure just skips the sidecar (the benchmark has built-in default
        names as a fallback).
    """
    import importlib.util

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "jetson", "fp16_mm2_layers.py")
    try:
        spec = importlib.util.spec_from_file_location("_fp16_mm2_layers", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        names = mod.island_node_names(src)
        with open(dst, "w") as f:
            f.write("# LiteMLA mm2 normalization nodes to pin FP32 in the FP16 engine\n")
            f.write("\n".join(names) + "\n")
        print(f"  fp16 mm2 sidecar -> {dst}  ({len(names)} node names)")
    except Exception as e:  # noqa: BLE001
        print(f"[onnx_qdq] WARNING: could not write fp16 mm2 sidecar ({e}); "
              f"benchmark_jetson.py will use its built-in defaults.", file=sys.stderr)



# ONNX export


def export_float_onnx(
    model: nn.Module,
    dummy: torch.Tensor,
    path: str,
    opset: int = 13,
    static: bool = False,
) -> str:
    """Export a BN-folded float model to ONNX (legacy TorchScript exporter).

        BN is folded before export so the graph only has Conv nodes (no BatchNorm), which is
        what inject_qdq expects. `model` (eval mode) is deep-copied, `dummy` is an example
        [B, C, H, W] tensor on CPU, opset 13 is safe for onnxruntime + TRT 8.4. Returns
        `path` so calls can be chained.

        static=True exports without dynamic axes. TRT 8.4 on the Jetson Xavier needs that:
        LiteMLA attention computes pad amounts and reshape shapes from H*W at runtime, and with
        dynamic spatial dims those become 2-dim shape-tensor Reshape ops that TRT 8.4 can't
        handle (IShuffleLayer on a shape tensor must have 0 or 1 reshape dims). With static H/W
        the tracer folds H*W%key_dim to 0 and those nodes disappear.
    """
    m = copy.deepcopy(model).cpu().eval()
    fold_bn(m)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dynamic_axes = (
        None
        if static
        else {"image": {0: "batch"}, "logits": {0: "batch"}}
    )
    torch.onnx.export(
        m, dummy.cpu().float(),
        path,
        opset_version=opset,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,          # legacy exporter -- stable, no NaN on batch 0
    )
    mode = "static" if static else "dynamic-batch"
    print(f"  float ONNX ({mode}) -> {path}  ({os.path.getsize(path) // 1024} KB)")
    return path



# QDQ node injection


def _make_init(name: str, arr: np.ndarray):
    """Wrap a numpy array as an ONNX initializer (TensorProto)."""
    from onnx import numpy_helper
    return numpy_helper.from_array(arr, name=name)


def _calibrate_matmul_scales(
    float_onnx_path: str,
    cal_batches: list,
) -> dict:
    """Per-tensor amax for every MatMul/Einsum input in the float ONNX.

        Runs the float ONNX through ORT with the MatMul/Einsum input tensors added as extra
        outputs and returns {tensor_name: amax / 127.0}, keyed by the exact input tensor names
        in the graph. Only activation inputs are included. Initializers are skipped, the
        weight-like constants in MatMul are already initializers and aren't dynamically
        quantized here.
    """
    import onnx
    import onnxruntime as ort
    import tempfile

    model = onnx.load(float_onnx_path)
    graph = model.graph

    init_names = {init.name for init in graph.initializer}
    # Tensors that are outputs of another MatMul/Einsum must not be quantized here. In LiteMLA
    # linear attention vk = mm1(v, k^T) is MatMul_output_0 and out = mm2(vk, q), so vk sits
    # between two MatMuls. Its last channel is the normalizer (sum of keys). An INT8 scale of
    # ~200-400 squashes that to 1-2 distinct values, the division turns into a step function
    # and attention collapses. Leaving out these tensors keeps mm1 in INT8 and mm2 in FP16,
    # which is the stable setup (TRT only runs mm2 in INT8 if both inputs are quantized, so
    # keeping vk in FP16 stops TRT from lowering mm2).
    matmul_output_names = {
        nd.output[0]
        for nd in graph.node
        if nd.op_type in ("MatMul", "Einsum") and nd.output
    }
    target_tensors = set()
    for node in graph.node:
        if node.op_type in ("MatMul", "Einsum"):
            for inp in node.input:
                if inp and inp not in init_names and inp not in matmul_output_names:
                    target_tensors.add(inp)

    if not target_tensors:
        print("[matmul_qdq] no MatMul/Einsum activation inputs found — nothing to calibrate")
        return {}

    # temporary model: add target tensors to the graph outputs
    tmp_model = onnx.ModelProto()
    tmp_model.CopyFrom(model)
    for name in target_tensors:
        vi = tmp_model.graph.output.add()
        vi.name = name

    amaxes = {name: 0.0 for name in target_tensors}

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    import os as _os
    onnx.save(tmp_model, tmp_path)

    try:
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        all_out_names = [o.name for o in tmp_model.graph.output]
        target_indices = [i for i, n in enumerate(all_out_names) if n in target_tensors]
        in_name = sess.get_inputs()[0].name

        for batch in cal_batches:
            results = sess.run(None, {in_name: batch["image"].float().numpy()})
            for idx in target_indices:
                name = all_out_names[idx]
                amax = float(np.abs(results[idx]).max())
                if amax > amaxes[name]:
                    amaxes[name] = amax
    finally:
        _os.unlink(tmp_path)

    scales = {name: max(amax / 127.0, 1e-8) for name, amax in amaxes.items()}
    print(f"[matmul_qdq] calibrated {len(scales)} MatMul/Einsum input tensors")
    for name, sc in sorted(scales.items()):
        print(f"  {name[-70:]:70s}  scale={sc:.6f}")
    return scales


def _inject_matmul_qdq(
    graph,
    matmul_input_scales: dict,
) -> list:
    """Inject per-tensor symmetric INT8 Q-DQ on MatMul/Einsum activation inputs.

        Modifies graph.node in-place: inserts a QL+DQL pair before each qualifying
        MatMul/Einsum input and rewires the node to consume the DQL output.

        TRT 8.4 constraints honoured:
          - No axis attribute (per-tensor QuantizeLinear only).
          - Zero-point is scalar INT8 zero (symmetric quantization).
          - No output Q-DQ on MatMul, only the inputs are wrapped.

        Returns list of new ONNX initializers that the caller must append to
        graph.initializer.
    """
    from onnx import helper, numpy_helper

    new_inits = []
    counter = 0
    new_nodes = []

    for node in graph.node:
        if node.op_type not in ("MatMul", "Einsum"):
            new_nodes.append(node)
            continue

        for slot in range(len(node.input)):
            inp = node.input[slot]
            if not inp or inp not in matmul_input_scales:
                continue

            scale = matmul_input_scales[inp]
            tag = f"_mm{counter}"
            counter += 1

            sc_name  = f"mm_scale{tag}"
            zp_name  = f"mm_zp{tag}"
            ql_out   = f"mm_ql{tag}"
            dql_out  = f"mm_dql{tag}"

            new_inits.append(
                numpy_helper.from_array(np.array(scale, dtype=np.float32), sc_name)
            )
            new_inits.append(
                numpy_helper.from_array(np.array(0, dtype=np.int8), zp_name)
            )
            # No axis attribute -> per-tensor QL (required for TRT 8.4 on Jetson Xavier)
            new_nodes.append(helper.make_node(
                "QuantizeLinear",
                inputs=[inp, sc_name, zp_name],
                outputs=[ql_out],
            ))
            new_nodes.append(helper.make_node(
                "DequantizeLinear",
                inputs=[ql_out, sc_name, zp_name],
                outputs=[dql_out],
            ))
            node.input[slot] = dql_out

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return new_inits


def _qdq_nodes_for_conv(
    conv_node,
    qconv: QuantConv2d,
    idx: int,
    float_init_map: dict,
    per_tensor_activation: bool = False,
    trt_compat: str = "jp50",
) -> Tuple[list, list, list]:
    """Build Q-DQ nodes and new initializers for one Conv.

        Args:
            per_tensor_activation: If True, collapse per-channel activation scales
                to a single per-tensor scalar (required for TRT GPU -- TRT does not
                support axis=1 QuantizeLinear on non-constant/dynamic tensors).
                If False, use per-channel axis=1 (onnxruntime supports this).
            trt_compat: TRT version compatibility mode for weight encoding:
                "jp50"   -- float32 weight initializer + QuantizeLinear + DequantizeLinear
                            (required for TRT 8.4.x / JetPack 5.0; bare INT8 initializers
                            cause a segfault in trtexec on Jetson Xavier).
                "modern" -- INT8 weight initializer + DequantizeLinear only
                            (supported by TRT 8.5+ / JetPack 5.1+ and desktop TRT >= 8.5;
                            produces a smaller ONNX file, roughly 4x less weight storage).

        Returns:
            pre_nodes  : list of ONNX nodes to insert BEFORE conv_node
            new_inits  : list of new TensorProto initializers to add to graph
            stale_names: names of float weight initializers that are now replaced
    """
    from onnx import helper

    tag = f"_c{idx}"
    pre_nodes:   list = []
    new_inits:   list = []
    stale_names: list = []

    # activation Q-DQ
    # Reconstruct symmetric scales from the asymmetric (scale, zp) pair stored
    # by manual_w8a8 calibration. Symmetric (zp=0) is required by TRT GPU;
    # onnxruntime accepts both forms.
    act_s_raw  = qconv._act_scale.numpy().astype(np.float64)  # [C_in]
    act_zp_raw = qconv._act_zp.numpy().astype(np.float64)     # [C_in], unclamped

    lo   = (-128.0 - act_zp_raw) * act_s_raw
    hi   = ( 127.0 - act_zp_raw) * act_s_raw
    amax = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1e-8)
    act_s_per_ch = (amax / 127.0).astype(np.float32)          # [C_in], per-channel

    if per_tensor_activation:
        # TRT GPU: collapse to single scalar via max -- conservative but valid.
        # Axis attribute must NOT be present for per-tensor QuantizeLinear.
        act_s  = np.array(act_s_per_ch.max(), dtype=np.float32)  # scalar
        act_zp = np.array(0, dtype=np.int8)                       # scalar
        ql_kwargs  = {}   # no axis -> per-tensor
        dql_kwargs = {}
    else:
        # onnxruntime: keep per-channel precision (axis=1 on [B,C,H,W]).
        act_s  = act_s_per_ch                                     # [C_in]
        act_zp = np.zeros(act_s.shape, dtype=np.int8)             # [C_in]
        ql_kwargs  = {"axis": 1}
        dql_kwargs = {"axis": 1}

    asc_name = f"act_scale{tag}"
    azp_name = f"act_zp{tag}"
    new_inits += [_make_init(asc_name, act_s), _make_init(azp_name, act_zp)]

    orig_x   = conv_node.input[0]
    ql_x_out = f"ql_x{tag}"
    dq_x_out = f"dq_x{tag}"

    pre_nodes.append(helper.make_node(
        "QuantizeLinear",
        inputs=[orig_x, asc_name, azp_name],
        outputs=[ql_x_out],
        **ql_kwargs,
    ))
    pre_nodes.append(helper.make_node(
        "DequantizeLinear",
        inputs=[ql_x_out, asc_name, azp_name],
        outputs=[dq_x_out],
        **dql_kwargs,
    ))
    conv_node.input[0] = dq_x_out

    # weight: per-channel symmetric Q-DQ (axis=0)
    w_name_orig = conv_node.input[1]
    w_s  = qconv.w_scale.numpy().astype(np.float32)    # [C_out]
    w_zp = np.zeros(w_s.shape, dtype=np.int8)           # symmetric -> zp = 0
    wsc_name = f"w_scale{tag}"
    wzp_name = f"w_zp{tag}"
    dq_w_out = f"dq_w{tag}"

    if trt_compat == "jp50":
        # JetPack 5.0 / TRT 8.4.x: bare INT8 initializer + DequantizeLinear
        # causes a segfault in trtexec on Jetson Xavier (sm_72).
        # Workaround: emit float32 weight -> QuantizeLinear -> DequantizeLinear.
        # ONNX file is ~4x larger but the compiled engine is identical.
        w_f32 = qconv.weight.detach().numpy().astype(np.float32)
        wf_name  = f"w_float{tag}"
        ql_w_out = f"ql_w{tag}"
        new_inits += [
            _make_init(wf_name,  w_f32),
            _make_init(wsc_name, w_s),
            _make_init(wzp_name, w_zp),
        ]
        pre_nodes.append(helper.make_node(
            "QuantizeLinear",
            inputs=[wf_name, wsc_name, wzp_name],
            outputs=[ql_w_out],
            axis=0,
        ))
        pre_nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=[ql_w_out, wsc_name, wzp_name],
            outputs=[dq_w_out],
            axis=0,
        ))
    else:
        # modern (TRT 8.5+ / JetPack 5.1+): INT8 initializer + DequantizeLinear.
        # Compact form: no extra QuantizeLinear node, weights pre-quantized to INT8.
        w_int8 = np.clip(
            np.round(qconv.weight.detach().numpy().astype(np.float32)
                     / w_s[:, None, None, None]),
            -128, 127,
        ).astype(np.int8)
        wi_name = f"w_int8{tag}"
        new_inits += [
            _make_init(wi_name,  w_int8),
            _make_init(wsc_name, w_s),
            _make_init(wzp_name, w_zp),
        ]
        pre_nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=[wi_name, wsc_name, wzp_name],
            outputs=[dq_w_out],
            axis=0,
        ))

    conv_node.input[1] = dq_w_out

    # original float weight initializer is now unreferenced
    if w_name_orig in float_init_map:
        stale_names.append(w_name_orig)

    return pre_nodes, new_inits, stale_names


def inject_qdq(
    float_onnx_path: str,
    quant_convs: List[QuantConv2d],
    output_path: str,
    skip_last_n: int = 1,
    per_tensor_activation: bool = False,
    with_output_qdq: bool = False,
    trt_compat: str = "jp50",
    matmul_scales: dict | None = None,
) -> str:
    """Inject Q-DQ nodes into a float ONNX graph.

        per_tensor_activation=False (default): per-channel activation Q-DQ (axis=1).
            Use for onnxruntime -- gives best accuracy (~0.014 drop, float Conv).
        per_tensor_activation=True: scalar activation Q-DQ (no axis attribute).
            Required for TensorRT GPU -- TRT does not support axis=1 on dynamic
            (non-constant) tensors.  Collapses per-channel scales to max.

        with_output_qdq=False (default): input-only Q-DQ.
            onnxruntime runs float32 Conv after DequantizeLinear -> 0.014 mIoU drop.
            Equivalent to manual_w8a8 but in ONNX form. Use for ORT evaluation.
        with_output_qdq=True: matched input+output Q-DQ pairs.
            Required for TRT explicit-quantization mode to form INT8 subgraphs.
            onnxruntime uses real INT8 Conv -> ~0.10 drop. Use for TRT deployment.

        Conv nodes in the ONNX graph (topological order) are matched 1-to-1 with
        quant_convs (same traversal order from _replace_convs in manual.py).
        The last skip_last_n Conv nodes are left float.

        Args:
            float_onnx_path : path to float ONNX (BN-folded, from export_float_onnx).
            quant_convs     : calibrated QuantConv2d list (skip_last_n already excluded).
            output_path     : where to write the QDQ ONNX.
            skip_last_n     : Conv nodes at the end to leave float (final head).
            trt_compat      : "jp50" (default) for TRT 8.4.x / JetPack 5.0 -- emits
                              float32 weights with full QL+DQL bracket.
                              "modern" for TRT 8.5+ / JetPack 5.1+ -- emits compact
                              INT8 initializer with DQL only.

        Returns:
            output_path.
    """
    import onnx
    from onnx import numpy_helper

    model_onnx = onnx.load(float_onnx_path)
    graph = model_onnx.graph

    conv_nodes  = [n for n in graph.node if n.op_type == "Conv"]
    n_total     = len(conv_nodes)
    n_quant     = n_total - skip_last_n

    if n_quant <= 0:
        raise ValueError(f"Nothing to quantize: {n_total} Conv nodes, skip_last_n={skip_last_n}")
    if len(quant_convs) != n_quant:
        raise ValueError(
            f"Mismatch: {len(quant_convs)} QuantConv2d layers "
            f"vs {n_quant} Conv nodes to quantize in ONNX.\n"
            f"Ensure skip_last_n={skip_last_n} is the same in both calls."
        )

    # Build initializer lookup for stale-name removal and sanity checks
    float_init_map = {init.name: init for init in graph.initializer}

    # Sanity check: weight shapes must agree
    for i, (cnode, qconv) in enumerate(zip(conv_nodes[:n_quant], quant_convs)):
        w_name = cnode.input[1]
        if w_name in float_init_map:
            onnx_shape = list(numpy_helper.to_array(float_init_map[w_name]).shape)
            pt_shape   = list(qconv.weight.shape)
            if onnx_shape != pt_shape:
                raise ValueError(
                    f"Conv[{i}] weight shape mismatch -- "
                    f"ONNX {onnx_shape} vs PyTorch {pt_shape}.\n"
                    "Check that the same BN-folded model was exported."
                )

    # Phase 1: inject input Q-DQ before each quantised Conv.
    # Collect output remapping for output Q-DQ (Phase 2).
    all_new_inits: list = []
    stale_init_names: set = set()
    quant_idx = 0
    output_remap: dict = {}   # old_conv_out_name -> dql_out_name
    output_qdq_nodes: list = []

    new_nodes: list = []
    for node in graph.node:
        if node.op_type == "Conv" and quant_idx < n_quant:
            qconv = quant_convs[quant_idx]
            if qconv._act_scale is not None:
                pre, inits, stale = _qdq_nodes_for_conv(
                    node, qconv, quant_idx, float_init_map,
                    per_tensor_activation=per_tensor_activation,
                    trt_compat=trt_compat,
                )
                new_nodes.extend(pre)
                all_new_inits.extend(inits)
                stale_init_names.update(stale)

            # Phase 2 prep: output Q-DQ when _out_scale is calibrated
            if with_output_qdq and hasattr(qconv, "_out_scale"):
                from onnx import helper as _h
                old_out  = node.output[0]
                tag      = f"_c{quant_idx}"
                raw_out  = f"conv_raw_out{tag}"   # Conv -> QL
                ql_out   = f"ql_out{tag}"          # QL -> DQL
                dql_out  = f"dql_out{tag}"         # DQL -> downstream

                node.output[0] = raw_out
                output_remap[old_out] = dql_out

                sc  = qconv._out_scale.numpy().astype(np.float32)   # scalar
                sc_name = f"out_scale{tag}"
                zp_name = f"out_zp{tag}"
                all_new_inits += [
                    _make_init(sc_name, sc),
                    _make_init(zp_name, np.array(0, dtype=np.int8)),
                ]
                # No axis attribute -> per-tensor quantization
                output_qdq_nodes.append(_h.make_node(
                    "QuantizeLinear",
                    inputs=[raw_out, sc_name, zp_name],
                    outputs=[ql_out],
                    name=f"QL_out{tag}",
                ))
                output_qdq_nodes.append(_h.make_node(
                    "DequantizeLinear",
                    inputs=[ql_out, sc_name, zp_name],
                    outputs=[dql_out],
                    name=f"DQL_out{tag}",
                ))
            quant_idx += 1
        new_nodes.append(node)

    # Phase 2: reroute all downstream consumers of remapped Conv outputs
    for node in new_nodes + output_qdq_nodes:
        for j, inp in enumerate(node.input):
            if inp in output_remap:
                node.input[j] = output_remap[inp]

    # Replace node list
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.node.extend(output_qdq_nodes)

    # Remove now-unreferenced float weight initializers (saves ~half the file size)
    surviving = [i for i in graph.initializer if i.name not in stale_init_names]
    del graph.initializer[:]
    graph.initializer.extend(surviving)
    graph.initializer.extend(all_new_inits)

    # Phase 3: MatMul/Einsum INT8 Q-DQ (TRT path only).
    # Calibration was run on the float ONNX before any QDQ injection so the
    # tensor names in matmul_scales match those still present in graph.node
    # after Phase 1+2 (Conv input/output QDQ renaming affects Conv outputs but
    # not the Pad/Transpose/Relu tensors that feed directly into MatMul inputs).
    if matmul_scales:
        mm_inits = _inject_matmul_qdq(graph, matmul_scales)
        graph.initializer.extend(mm_inits)
        n_mm_inputs = len(mm_inits) // 2
        print(f"  MatMul/Einsum QDQ: {n_mm_inputs} input tensors quantised "
              f"(expect 16 for 8 MatMuls × 2 inputs)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model_onnx, output_path)

    n_stale_kb = sum(
        np.prod(numpy_helper.to_array(float_init_map[n]).shape) * 4 // 1024
        for n in stale_init_names if n in float_init_map
    )
    print(
        f"  QDQ ONNX -> {output_path}  ({os.path.getsize(output_path) // 1024} KB)\n"
        f"  {quant_idx}/{n_total} Conv quantised  |  "
        f"removed {len(stale_init_names)} float weight tensors (~{n_stale_kb} KB freed)"
    )
    return output_path



# Main entry point


def build_qdq_onnx(
    float_model: nn.Module,
    cal_batches: list,
    out_dir: str,
    skip_last_n: int = 1,
    opset: int = 13,
    smooth_quant_alpha: float = 0.5,
    trt_compat: str = "jp50",
    save_calib_images: bool = False,
    matmul_qdq: bool = False,
) -> Tuple[str, str, str]:
    """Full pipeline: calibrate -> export float ONNX -> inject QDQ.

        Pipeline:
          1. Calibrate per-channel INT8 scales (manual_w8a8 pipeline).
          2. Export BN-folded float model to ONNX.
          3. Calibrate per-tensor output scales (99.99th-percentile).
          4. Inject Q-DQ nodes:
               model_qdq.onnx     -- per-channel input Q-DQ, no output Q-DQ
                                     (onnxruntime float Conv path; ~0.014 drop)
               model_qdq_trt.onnx -- per-tensor input+output Q-DQ
                                     (TRT explicit INT8 subgraph)

        Args:
            float_model        : trained float32 model in eval mode.
            cal_batches        : list of {'image': Tensor, ...} for calibration.
            out_dir            : directory for output ONNX files.
            skip_last_n        : Conv layers to keep float (default 1 = head).
            opset              : ONNX opset (13 recommended).
            smooth_quant_alpha : SmoothQuant migration strength (default 0.5, unused here).
            trt_compat         : "jp50" or "modern" (see inject_qdq).

        Returns:
            (float_onnx_path, qdq_onnx_path, qdq_trt_onnx_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: calibrate per-channel scales via manual_w8a8
    print("[onnx_qdq] 1/3  calibrate per-channel scales (manual_w8a8)")
    quant_model = quantize_model(float_model, cal_batches, skip_last_n=skip_last_n)
    quant_convs = [
        m for _, m in quant_model.named_modules()
        if isinstance(m, QuantConv2d)
    ]
    print(f"  {len(quant_convs)} QuantConv2d layers calibrated")

    # Step 2: export BN-folded float ONNX + calibrate output scales
    print("[onnx_qdq] 2/3  export float ONNX + calibrate output scales")
    dummy = cal_batches[0]["image"][:1].float().cpu()
    float_path = os.path.join(out_dir, "model_float.onnx")
    export_float_onnx(quant_model, dummy, float_path, opset)

    # jp50/TRT 8.4: also export a fully static ONNX and run onnxsim to fold
    # LiteMLA's Shape->Gather->Mul shape-tensor chains. TRT 8.4 rejects
    # IShuffleLayer on shape tensors with >1 dim; onnxsim eliminates them.
    if trt_compat == "jp50":
        float_static_path = os.path.join(out_dir, "model_float_static.onnx")
        float_trt_path    = os.path.join(out_dir, "model_float_trt_base.onnx")
        export_float_onnx(quant_model, dummy, float_static_path, opset, static=True)
        _trt_base_ok = _simplify_onnx(float_static_path, float_trt_path,
                                       input_shapes={"image": list(dummy.shape)})
        _trt_base_exists = _trt_base_ok and os.path.getsize(float_trt_path) > 0
        # Sidecar listing the LiteMLA mm2 normalization node names. benchmark_jetson.py
        # reads it to pin those layers to FP32 in the FP16 engine (FP32 accumulation
        # for the attention normalizer) via trtexec builder flags - see --fp16_mm2_fp32.
        # FP32/INT8 engines are unaffected.
        if _trt_base_exists:
            _write_fp16_mm2_sidecar(
                float_trt_path, os.path.join(out_dir, "fp16_mm2_layers.txt"))
    else:
        float_trt_path    = float_path
        _trt_base_exists  = False

    calibrate_output_scales(quant_model, cal_batches)

    # ORT float-Conv variant: per-channel input Q-DQ only -> accuracy reference.
    # No MatMul QDQ: ORT does not benefit from INT8 MatMul the same way TRT does,
    # and INT8 Einsum/MatMul support in ORT is incomplete.
    qdq_path = os.path.join(out_dir, "model_qdq.onnx")
    inject_qdq(float_path, quant_convs, qdq_path, skip_last_n,
               per_tensor_activation=False, with_output_qdq=False,
               trt_compat=trt_compat,
               matmul_scales=None)

    # Calibrate MatMul/Einsum scales from the onnxsim-simplified float ONNX.
    # Must use float_trt_path (not float_path) because the TRT injection reads
    # from float_trt_path; tensor names must match. Only done when requested.
    trt_src = float_trt_path if _trt_base_exists else float_path
    mm_scales: dict | None = None
    if matmul_qdq:
        n_steps = "3b"
        print(f"[onnx_qdq] {n_steps}/3  calibrate MatMul/Einsum input scales "
              f"(PTQ, {len(cal_batches)} batches)")
        mm_scales = _calibrate_matmul_scales(trt_src, cal_batches)

    # TRT variant: per-tensor input+output Q-DQ + optional MatMul QDQ.
    print(f"[onnx_qdq] 3/3  inject TRT Q-DQ nodes (per-tensor, trt_compat={trt_compat!r})")
    qdq_trt_path = os.path.join(out_dir, "model_qdq_trt.onnx")
    inject_qdq(trt_src, quant_convs, qdq_trt_path, skip_last_n,
               per_tensor_activation=True, with_output_qdq=True,
               trt_compat=trt_compat,
               matmul_scales=mm_scales)
    print(f"  TRT QDQ ONNX -> {qdq_trt_path}")

    if save_calib_images:
        calib_path = os.path.join(out_dir, "calib_images.npy")
        print(f"  saving {len(cal_batches)} calibration batches to {calib_path} …", flush=True)
        calib_arr = np.concatenate([b["image"].float().numpy() for b in cal_batches], axis=0)
        np.save(calib_path, calib_arr)
        print(f"  calibration images saved -> {calib_path}  ({calib_arr.shape})")

    return float_path, qdq_path, qdq_trt_path



# Evaluation and latency helpers (onnxruntime)


def eval_onnx(
    onnx_path: str,
    batches: list,
    num_classes: int,
    ignore_index: int = -1,
) -> dict:
    """Evaluate any ONNX model on pre-loaded batches. Returns metrics dict.

        Uses CUDA EP if available, falls back to CPU.
        batches: list of {'image': float Tensor, 'mask': long Tensor}.
    """
    import onnxruntime as ort
    from utils.metrics import accumulate_metrics, finalize_metrics

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    in_name = sess.get_inputs()[0].name

    conf = None
    for batch in batches:
        logits = torch.from_numpy(
            sess.run(None, {in_name: batch["image"].float().numpy()})[0]
        )
        conf = accumulate_metrics(logits, batch["mask"].long(), num_classes, ignore_index, conf)

    return finalize_metrics(conf) if conf is not None else {}


def latency_ms_onnx(
    onnx_path: str,
    batches: list,
    n_runs: int = 30,
    warmup: int = 5,
) -> float:
    """Mean per-batch CPU latency for an ONNX model (milliseconds)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    times = []
    for i, batch in enumerate(batches):
        if i >= n_runs + warmup:
            break
        img = batch["image"].float().numpy()
        if i < warmup:
            sess.run(None, {in_name: img})
            continue
        t0 = time.perf_counter()
        sess.run(None, {in_name: img})
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)) if times else 0.0
