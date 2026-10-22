# Supplement to device_package_20261020T160617Z (22 Oct 2026)

Adds the fully 8-bit (8-bit input and output) R4 models for seeds 42 and 91, so all four seeds can be
built and measured in every precision (32-bit, 16-bit, mixed, fully 8-bit). The original package is unchanged.

These two models FAILED the workstation CPU acceptance check (PROTOCOL.json `accuracy_tolerance`), marginally,
on the pooled mIoU criterion (limit: drop <= 0.01) against the 32-bit parent, Sen1Floods11 val, 89 images:

| seed | pooled mIoU drop | equal-event mIoU drop | pooled water IoU drop | CPU result |
|---|---|---|---|---|
| 42 | 0.0107 | 0.0071 | 0.0194 | rejected (pooled mIoU) |
| 91 | 0.0103 | 0.0091 | 0.0186 | rejected (pooled mIoU) |
| 64 (package) | 0.0067 | 0.0057 | 0.0121 | accepted |
| 107 (package) | 0.0047 | 0.0029 | 0.0084 | accepted |

Source: `evidence/R4_s{seed}_PURE_CPU.json`, `comparisons.integer_vs_float_parent`.
The files are byte copies of `quant_author_tools/<run>/pure_int8_io.onnx` and `PURE_IO.json`; hashes in `SHA256SUMS`.
`device_session.py` skips non-admitted models, so these must be built with `integer_tools.py build` using the same
arguments as the seed-64/107 pure models. Their device result is reported as measured; it does not override the CPU verdict
unless it is also within tolerance on the device.
