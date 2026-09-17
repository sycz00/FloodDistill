#!/usr/bin/env python3
"""Fix model_qdq_trt.onnx for TRT 8.4 on the Jetson Xavier.

TRT 8.4 rejects two patterns in the LiteMLA attention blocks.

  1. INT64 "shape tensor" chain:
       Concat_2(INT64) -> Reshape_1([-1,2]) -> Slice -> Transpose -> Reshape_2([-1]) -> Pad
     TRT treats INT64 tensors as shape tensors and only allows 0D/1D reshapes on them.
     Fix: precompute the final 1D padding vector (every block gives [0,0,0,0,0,0,1,0]) and
     replace Reshape_2_output_0 with a frozen Constant, so the whole chain becomes dead.

  2. Dynamic batch dimension:
     The model input has a `dyn:batch` dim, so every Concat-built shape tensor downstream
     contains -1. Fix: freeze the input to [1,6,512,512], resolve the -1 dims in the Reshape
     shape tensors from data tensor sizes captured with ORT, then freeze the output to
     [1,2,512,512].

Usage:
    python3 jetson/fix_qdq_onnx.py [model_qdq_trt.onnx] [--out model_qdq_trt_static.onnx]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.helper as helper
import onnxruntime as ort



# Helpers


def _make_int64_const(name: str, values: List[int]) -> onnx.TensorProto:
    arr = np.array(values, dtype=np.int64)
    return helper.make_tensor(name, onnx.TensorProto.INT64, list(arr.shape), arr.tolist())


def _run_ort(model: onnx.ModelProto,
             extra_outputs: Dict[str, int],
             input_shape=(1, 6, 512, 512)) -> Dict[str, np.ndarray]:
    """Add extra_outputs to graph, run ORT on a dummy input, return all outputs."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    existing = {o.name for o in m.graph.output}
    for name, elem_type in extra_outputs.items():
        if name not in existing:
            vi = helper.make_tensor_value_info(name, elem_type, None)
            m.graph.output.append(vi)
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    dummy = np.zeros(input_shape, dtype=np.float32)
    outs = sess.run(None, {sess.get_inputs()[0].name: dummy})
    return dict(zip([o.name for o in sess.get_outputs()], outs))


def _add_frozen_initializer(model: onnx.ModelProto, tensor: onnx.TensorProto) -> None:
    model.graph.initializer.append(tensor)


def _replace_node_input(model: onnx.ModelProto, old: str, new: str) -> int:
    """Replace every occurrence of `old` as a node input with `new`. Returns count."""
    replaced = 0
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old:
                node.input[i] = new
                replaced += 1
    return replaced



# Pass 0 - cubic Resize to linear (TRT 8.4 has no cubic support)


def patch_cubic_resize(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace mode='cubic' with mode='linear' on all Resize nodes."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    patched = 0
    for node in m.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "mode" and attr.s == b"cubic":
                attr.s = b"linear"
                patched += 1
    if patched:
        print(f"  Patched {patched} cubic Resize node(s) → linear.")
    else:
        print("  No cubic Resize nodes found.")
    return m



# Pass 1 - freeze INT64 Pad-padding chain outputs


def _find_int64_pad_chain_outputs(model: onnx.ModelProto) -> List[str]:
    """Find the Reshape_2_output_0 tensors that feed Pad nodes as padding input.

        They come out of the INT64 chain
            Concat_2(INT64) -> Reshape_1([-1,2]) -> Slice -> Transpose -> Reshape_2([-1]) -> Pad
    """
    result = []
    for node in model.graph.node:
        if node.op_type != "Pad":
            continue
        if len(node.input) < 2:
            continue
        pads_input = node.input[1]
        result.append(pads_input)
    return list(dict.fromkeys(result))


def freeze_int64_pad_chains(model: onnx.ModelProto,
                             input_shape=(1, 6, 512, 512)) -> onnx.ModelProto:
    """Replace all Pad padding inputs with precomputed INT64 constants."""
    pad_inputs = _find_int64_pad_chain_outputs(model)
    if not pad_inputs:
        print("  No dynamic Pad padding inputs found.")
        return model

    print(f"  Found {len(pad_inputs)} Pad padding input(s). Capturing with ORT …")
    extra = {name: onnx.TensorProto.INT64 for name in pad_inputs}
    ort_out = _run_ort(model, extra, input_shape)

    m = onnx.ModelProto()
    m.CopyFrom(model)

    for orig_name in pad_inputs:
        if orig_name not in ort_out:
            print(f"    WARNING: could not capture {orig_name!r}", file=sys.stderr)
            continue
        val = ort_out[orig_name].astype(np.int64).flatten()
        frozen_name = orig_name + "__frozen"
        tensor = _make_int64_const(frozen_name, val.tolist())
        _add_frozen_initializer(m, tensor)
        count = _replace_node_input(m, orig_name, frozen_name)
        print(f"    {orig_name.rsplit('/', 1)[-1]!r}  →  {val.tolist()}  ({count} consumer(s))")

    return m



# Pass 2 - freeze dynamic Reshape shape inputs and resolve -1 dims


def _collect_dynamic_reshape_info(model: onnx.ModelProto) -> List[Tuple[str, str, str]]:
    """Return (reshape_node_name, data_input_name, shape_input_name) for dynamic Reshapes."""
    init_names = {i.name for i in model.graph.initializer}
    result = []
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        if len(node.input) < 2:
            continue
        shape_in = node.input[1]
        if shape_in and shape_in not in init_names:
            result.append((node.name, node.input[0], shape_in))
    return result


def _resolve_minus1(shape: List[int], data_shape: Tuple) -> List[int]:
    """Replace -1 in shape with the inferred value given data_shape."""
    total = 1
    for d in data_shape:
        total *= d
    known = 1
    for d in shape:
        if d != -1:
            known *= d
    return [total // known if d == -1 else d for d in shape]


def freeze_reshape_shape_inputs(model: onnx.ModelProto,
                                 input_shape=(1, 6, 512, 512)) -> onnx.ModelProto:
    """Freeze all dynamic Reshape shape inputs, resolving -1 dims to concrete values."""
    dynamic_info = _collect_dynamic_reshape_info(model)
    if not dynamic_info:
        print("  No dynamic Reshape shape inputs remaining.")
        return model

    print(f"  Found {len(dynamic_info)} dynamic Reshape shape input(s). Capturing with ORT …")

    # Need: the shape input values (INT64) and the data input shapes (FLOAT or INT64)
    shape_names = list(dict.fromkeys([si for _, _, si in dynamic_info]))
    data_names  = list(dict.fromkeys([di for _, di, _ in dynamic_info]))

    extra_shape = {n: onnx.TensorProto.INT64 for n in shape_names}
    extra_data  = {n: onnx.TensorProto.FLOAT for n in data_names}

    # Capture shape values
    try:
        ort_out = _run_ort(model, {**extra_shape, **extra_data}, input_shape)
    except Exception:
        # Some data inputs might be INT64; try without them
        ort_out = _run_ort(model, extra_shape, input_shape)

    m = onnx.ModelProto()
    m.CopyFrom(model)
    init_names = {i.name for i in m.graph.initializer}

    for node_name, data_in, shape_in in dynamic_info:
        if shape_in in init_names:
            continue  # already frozen by a previous call
        if shape_in not in ort_out:
            print(f"    WARNING: no ORT value for shape input {shape_in!r}", file=sys.stderr)
            continue

        raw_shape = ort_out[shape_in].astype(np.int64).flatten().tolist()

        # Resolve -1 if data tensor shape is available
        if -1 in raw_shape and data_in in ort_out:
            data_shape = ort_out[data_in].shape
            concrete_shape = _resolve_minus1(raw_shape, data_shape)
        else:
            concrete_shape = raw_shape  # keep -1 for TRT to infer (valid for float tensors)

        frozen_name = shape_in + "__frozen"
        tensor = _make_int64_const(frozen_name, concrete_shape)
        _add_frozen_initializer(m, tensor)
        count = _replace_node_input(m, shape_in, frozen_name)
        suffix = shape_in.rsplit("/", 1)[-1]
        print(f"    {suffix!r}  {raw_shape} → {concrete_shape}  ({count} consumer(s))")
        init_names.add(frozen_name)

    return m



# Pass 2b - prune dead nodes


def prune_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove nodes whose outputs are never consumed (dead code after constant folding)."""
    m = onnx.ModelProto()
    m.CopyFrom(model)

    graph_output_names = {o.name for o in m.graph.output}
    init_names = {i.name for i in m.graph.initializer}

    rounds = 0
    while True:
        consumed: set = set(graph_output_names)
        for node in m.graph.node:
            for inp in node.input:
                if inp:
                    consumed.add(inp)

        dead = [node for node in m.graph.node
                if all(o not in consumed or o == "" for o in node.output)]

        if not dead:
            break
        rounds += 1
        dead_names = {o for node in dead for o in node.output if o}
        kept = [node for node in m.graph.node if node not in dead]
        del m.graph.node[:]
        m.graph.node.extend(kept)
        # Also remove dead initializers that were only used by dead nodes
        dead_inits = [i for i in m.graph.initializer
                      if i.name in dead_names and i.name not in consumed]
        for di in dead_inits:
            m.graph.initializer.remove(di)

    if rounds:
        print(f"  Pruned dead nodes in {rounds} pass(es).")
    return m



# Pass 3 - make activation QL/DQL zero-points scalar for ORT compatibility


def fix_activation_zero_points(model: onnx.ModelProto) -> onnx.ModelProto:
    """Collapse per-channel activation ZPs to a scalar so ORT can fuse QDQ+Conv.

        ORT's internal QLinearConv fusion needs per-tensor activation QuantizeLinear /
        DequantizeLinear nodes (no axis, scalar ZP), but some QAT exporters emit per-channel
        activation quantizers by mistake. Activation QL/DQL nodes are the ones whose input[0]
        is a runtime tensor and not a weight initializer, recognised by a ZP name with the
        "act_" prefix or by a per-channel ZP next to a scalar scale. For those the 'axis'
        attribute is removed (per-tensor) and input[2] is pointed at a shared scalar-0
        initializer. Only that node's input changes, there's no global rename, so weight DQL
        nodes that might share ZP tensors don't break. With symmetric QAT all activation ZPs
        are 0, so the collapse is always correct.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    init_names = {i.name for i in m.graph.initializer}
    init_map   = {i.name: i for i in m.graph.initializer}

    # A single scalar-0 INT8 zero-point shared by all fixed activation nodes.
    SCALAR_NAME = "__act_zp_scalar0"
    if SCALAR_NAME not in init_map:
        scalar_t = helper.make_tensor(SCALAR_NAME, onnx.TensorProto.INT8, [], [0])
        m.graph.initializer.append(scalar_t)
        init_map[SCALAR_NAME] = scalar_t

    # In jp50-format QDQ the weights go float, then QL, then DQL. The weight QL has
    # input[0] = w_float (a static initializer) and outputs ql_w_c{i}. The weight DQL then
    # has input[0] = ql_w_c{i}, which is not an initializer, so the guard below would miss
    # it. Collect all weight QL output names so the weight DQL gets skipped as well.
    weight_ql_outputs = {
        n.output[0]
        for n in m.graph.node
        if n.op_type == "QuantizeLinear"
        and n.input and n.input[0] in init_names
    }

    patched = 0
    for node in m.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue

        # Weight quantizers: skip both the weight QL (input[0] is an initializer)
        # and the weight DQL (input[0] is the ql_w output of the weight QL).
        if node.input and (node.input[0] in init_names
                           or node.input[0] in weight_ql_outputs):
            continue

        if len(node.input) < 3 or not node.input[2]:
            continue

        zp = init_map.get(node.input[2])
        if zp is None or zp.data_type not in (
                onnx.TensorProto.INT8, onnx.TensorProto.UINT8):
            continue

        dtype_np = np.int8 if zp.data_type == onnx.TensorProto.INT8 else np.uint8
        vals = (np.frombuffer(zp.raw_data, dtype=dtype_np) if zp.raw_data
                else np.array(list(zp.int32_data), dtype=dtype_np))

        if vals.size <= 1:
            continue  # already scalar - also remove stray axis attribute below

        if not np.all(vals == vals[0]):
            print(f"    WARNING: non-uniform activation ZP {node.input[2]!r} — skipping")
            continue

        # Remove axis attribute (makes it per-tensor) and point ZP to scalar.
        for attr in list(node.attribute):
            if attr.name == "axis":
                node.attribute.remove(attr)
        node.input[2] = SCALAR_NAME
        patched += 1

    if patched:
        print(f"  Collapsed {patched} activation ZP(s) → scalar (per-tensor).")
    else:
        print("  No non-scalar activation ZPs found — model already ORT-compatible.")
    return m



# Pass 4 - freeze I/O shapes (called after zero-point fix)


def freeze_io_shapes(model: onnx.ModelProto,
                     input_shape=(1, 6, 512, 512),
                     output_shape=(1, 2, 512, 512)) -> onnx.ModelProto:
    m = onnx.ModelProto()
    m.CopyFrom(model)

    def _set(vi, shape):
        t = vi.type.tensor_type
        t.shape.ClearField("dim")
        for d in shape:
            dim = t.shape.dim.add()
            dim.dim_value = d

    for inp in m.graph.input:
        if inp.name == "image":
            _set(inp, input_shape)
    for out in m.graph.output:
        if out.name == "logits":
            _set(out, output_shape)
    return m



# Main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?",
                    default="runs/deploy_qat/docker_qat/model_qdq_trt_no_out.onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    in_path  = Path(args.input).expanduser().resolve()
    out_path = (Path(args.out).expanduser().resolve() if args.out
                else in_path.parent / (in_path.stem + "_static.onnx"))

    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    input_shape  = (args.batch, args.channels, args.img_size, args.img_size)
    output_shape = (args.batch, 2, args.img_size, args.img_size)

    print(f"Loading {in_path} …")
    model = onnx.load(str(in_path))

    print(f"\n[Pass 0] Patch cubic Resize → linear …")
    model = patch_cubic_resize(model)

    print(f"\n[Pass 1] Freeze INT64 Pad-padding chain …")
    model = freeze_int64_pad_chains(model, input_shape)

    print(f"\n[Pass 2] Freeze dynamic Reshape shape inputs …")
    model = freeze_reshape_shape_inputs(model, input_shape)

    print(f"\n[Pass 2b] Prune dead nodes …")
    model = prune_dead_nodes(model)

    print(f"\n[Pass 3] Collapse per-channel activation zero-points to scalar …")
    model = fix_activation_zero_points(model)

    print(f"\n[Pass 4] Freeze I/O shapes {input_shape} → {output_shape} …")
    model = freeze_io_shapes(model, input_shape, output_shape)

    print(f"\nSaving to {out_path} …")
    onnx.save(model, str(out_path))
    print(f"  {out_path.stat().st_size / 1e6:.1f} MB")

    print("\nVerifying with ORT …")
    try:
        dummy = np.zeros(input_shape, dtype=np.float32)
        sess  = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        out   = sess.run(None, {sess.get_inputs()[0].name: dummy})
        expected = output_shape
        got = out[0].shape
        status = "PASS" if got == expected else "MISMATCH"
        print(f"  {status} — output shape: {got}  (expected {expected})")
        if got != expected:
            sys.exit(1)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
