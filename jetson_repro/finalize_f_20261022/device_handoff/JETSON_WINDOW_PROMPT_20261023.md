Use the new 23-October `device_handoff` in a fresh directory beside the old checkout. Preserve local commits, old attempts and failures. Keep the existing flood Python/TensorRT8.4 environment, clocks and power mode.

Fixes: B0 mixed uses PREFER with mandatory FP32 attention/classifier audits; four aligned16 micro pure candidates replace the failed originals; STURM image hashes are informational: accept OpenCV-version upsampling differences within absolute 1e-4 in normalized inputs against the float64 bilinear reference; keep mask checks. Native recovery still needs measurement.

Find the current full-matrix `results/attempt` and resolve the paths below; do not use a smoke folder. Link its exact 353 prepared arrays, verify, then queue:

```bash
cd "$NEW_HANDOFF"
"$JETSON_PYTHON" prepare_inputs.py --bundle-root "$OLD_HANDOFF" --link
sha256sum -c SHA256SUMS
ATTEMPT="followup_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results_logs
setsid "$JETSON_PYTHON" runtime/chain_evaluations.py \
  --predecessor "$OLD_ATTEMPT" --attempt "$ATTEMPT" \
  --sturm-root "$ACTUAL_STURM_ROOT" --wf-root "$ACTUAL_WF_ROOT" \
  </dev/null >"results_logs/$ATTEMPT.chain.log" 2>&1 &
```

If a standalone micro attempt already started, add another `--predecessor /actual/results/attempt`; replace any unstarted old micro queue with this follow-up. The chain waits for predecessors to finish, locks out old resumes, copies verified compatible completed results, and runs changed/missing cells. Keep old work progressing: PAUSE alone does not release its lock. Inspect `results_logs/$ATTEMPT.chain.json`, then the new attempt's HEARTBEAT/STATUS/CHAIN_IMPORT.

Retain 48 slots / 51 entries, every dataset/dry set/benchmark, and all four seeds. Reject floating micro pure engines, failed B0 FP32 audits and any source loss above 0.02. Keep strict export failures separate. Return complete results/logs and collection tables, disclosing inherited sessions/input profiles. Never overwrite old attempts, overlap heavy jobs, retrain, recalibrate, change masks, install software or commit/push results. Read JETSON_PROMPT.md for controls and full collection requirements.

If no controller attempt exists, start runtime/run_remaining.py with a fresh --attempt and the same dataset roots after confirming the device is idle; preserve smoke logs. A preflight-failed controller attempt with locks is a valid predecessor even if it has no model STATUS yet.
