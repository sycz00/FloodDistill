#!/usr/bin/env python3
"""List the LiteMLA mm2 normalization nodes that get pinned to FP32.

The FP16 TensorRT engine is built from the float ONNX with `trtexec --fp16` (TRT 8.4
miscompiles a pre-converted fp16 ONNX and attention collapses). To keep the LiteMLA
attention normalizer accumulating in FP32, the normalization sub-graph is pinned to FP32
with trtexec builder flags:

    MatMul_1 (mm2) -> Slice(numerator) / Slice(normalizer) -> Add(+eps) -> Div

mm2 gives the un-normalized attention output, and its numerator reaches ~6.5e4 (the FP16
limit) before the Div brings it back to O(1-10). So the whole region has to stay FP32, not
only mm2, or the numerator overflows when it's cast to FP16.

The node names are taken from the graph structure (walking forward from each
`.../main/MatMul_1` up to and including its Div) and written to a sidecar
`fp16_mm2_layers.txt` that benchmark_jetson.py reads (`--fp16_mm2_fp32`). Needs only `onnx`.
No fp16 conversion or onnxconverter_common.

Usage:
    python jetson/fp16_mm2_layers.py runs/deploy_qat/docker_qat/model_float_trt_base.onnx \
        --out runs/quant/<run>/fp16_mm2_layers.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx


MM_SUFFIXES = ("/context_module/main/MatMul", "/context_module/main/MatMul_1")


def island_node_names(onnx_path: str) -> list[str]:
    """Return the LiteMLA attention MatMul node names (mm1 and mm2 per block).

        Only the matmuls are listed: they keep their exact ONNX node names in the built
        TRT engine and so can be pinned by name, whereas the normalizer Add+Div fuse
        into a comma-named POINTWISE that trtexec --layerPrecisions cannot address.
    """
    model = onnx.load(str(onnx_path))
    return [n.name for n in model.graph.node
            if n.op_type == "MatMul" and n.name.endswith(MM_SUFFIXES)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="float ONNX (e.g. model_float_trt_base.onnx)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <input dir>/fp16_mm2_layers.txt)")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)
    out_path = (Path(args.out).expanduser().resolve() if args.out
                else in_path.parent / "fp16_mm2_layers.txt")

    names = island_node_names(str(in_path))
    if not names:
        print(f"ERROR: no LiteMLA attention MatMuls (names ending in {MM_SUFFIXES}) "
              f"in {in_path}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w") as f:
        f.write("# LiteMLA attention MatMuls (mm1, mm2) to pin FP32 in the FP16 engine\n")
        f.write("\n".join(names) + "\n")
    print(f"Wrote {len(names)} LiteMLA MatMul node name(s) -> {out_path}")
    for n in names:
        print(f"    {n.split('/backbone/')[-1]}")


if __name__ == "__main__":
    main()
