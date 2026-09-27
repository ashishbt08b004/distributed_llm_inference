#!/usr/bin/env bash
# Phase 8 — ablation grid: configs x arrival patterns (run ON the node).
#
#   ./08_run_benchmarks.sh quick          # ~5 min: the ACTIVE config only, no redeploy
#   ./08_run_benchmarks.sh quick B        # one more piece: switch to B (+ pod restart), ~5 min of runs
#   ./08_run_benchmarks.sh                # full grid A B C, every pattern (hours)
#
#   CONFIGS="B C" ./08_run_benchmarks.sh          # subset of configs
#   PATTERNS="multiturn" ./08_run_benchmarks.sh   # subset of patterns
#   BENCH_VIA_GATEWAY=true ./08_run_benchmarks.sh # B/C through LiteLLM instead of the router
#
# Each run writes results_<cfg>_<pattern>.json plus metrics snapshots taken
# before/after, into $BENCH_DIR/results (quick: $BENCH_DIR/results-quick, so the
# two profiles never mix). analyze.py turns a directory into summary.md, and
# re-running it after each piece adds that piece to the same table.
source "$(dirname "$0")/stack.sh"

PROFILE="full"
if [ "${1:-}" = "quick" ] || [ "${1:-}" = "full" ]; then PROFILE="$1"; shift; fi
[ $# -gt 0 ] && CONFIGS="$*"
ACTIVE="$(cat "$BP1_ROOT/.active_config" 2>/dev/null || true)"

if [ "$PROFILE" = "quick" ]; then
  # Same patterns in miniature: a paced run, a burst (exercises the admission
  # queue), and multi-turn (the prefix-cache / Mooncake story). rate1 is dropped:
  # it is the slowest pattern and the least informative.
  BENCH_RATES="${QUICK_RATES:-4 inf}"
  BENCH_NUM_PROMPTS="${QUICK_NUM_PROMPTS:-100}"
  MT_SESSIONS="${QUICK_MT_SESSIONS:-20}"
  MT_TURNS="${QUICK_MT_TURNS:-5}"
  RUN_SHAREGPT=false
  CONFIGS="${CONFIGS:-${ACTIVE:-C}}"    # default: whatever is deployed now
  RESULT_SUBDIR=results-quick
else
  CONFIGS="${CONFIGS:-$BENCH_CONFIGS}"
  RESULT_SUBDIR=results
fi
# Run the already-deployed config first: switching costs a pod restart.
if [ -n "$ACTIVE" ] && [[ " $CONFIGS " == *" $ACTIVE "* ]]; then
  CONFIGS="$ACTIVE $(for c in $CONFIGS; do if [ "$c" != "$ACTIVE" ]; then printf '%s ' "$c"; fi; done)"
fi
PATTERNS="${PATTERNS:-$(for r in $BENCH_RATES; do printf 'rate%s ' "$r"; done)multiturn$([ "$RUN_SHAREGPT" = "true" ] && echo ' sharegpt' || true)}"
RESULTS="/bench/$RESULT_SUBDIR"   # path inside bench-client (= $BENCH_DIR/$RESULT_SUBDIR on the host)
OUT="$BENCH_DIR/$RESULT_SUBDIR"
mkdir -p "$OUT"
log "Profile $PROFILE: configs [$CONFIGS], patterns [$PATTERNS] -> $OUT"
T_START=$SECONDS

log "Preparing bench-client pod"
kubectl -n inference create configmap bench-scripts --from-file="$BP1_ROOT/bench" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
apply_tpl "$BP1_ROOT/manifests/bench/bench-client.yaml" >/dev/null
wait_rollout inference deploy/bench-client 300 >/dev/null
BX=(kubectl -n inference exec deploy/bench-client --)

if [ "$RUN_SHAREGPT" = "true" ] && [ ! -f "$BENCH_DIR/ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
  log "Downloading ShareGPT dataset"
  wget -q -O "$BENCH_DIR/ShareGPT_V3_unfiltered_cleaned_split.json" \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

# Label results by what was actually deployed. Config C is defined with the KV tier;
# C without it is a different stack, so its results are kept apart as C-nokv.
result_label() {
  if [ "$1" = "C" ] && [ "${ENABLE_KV_TIER_C:-true}" != "true" ]; then echo "C-nokv"; else echo "$1"; fi
}

run_pattern() {
  local cfg="$1" pattern="$2" ep="$3" key="$4"; shift 4
  local targets=("$@")
  local tag; tag="$(result_label "$cfg")_${pattern}"
  local prev="$OUT/results_${tag}.json"
  if [ -f "$prev" ] && [ "${FORCE:-false}" != "true" ]; then
    # Reuse a result only if it is healthy AND was measured on the stack that is
    # deployed now (.active_config is rewritten by every pool deploy: Phase 5, set_config, Phase 6).
    if ! jq -e '(.completed // 0) > 0 and (.failed // 0) == 0 and (.completed >= (.num_prompts // .completed))' \
         "$prev" >/dev/null 2>&1; then
      warn "re-running $tag: the previous result has failed or missing requests"
    elif [ -f "$BP1_ROOT/.active_config" ] && [ "$prev" -ot "$BP1_ROOT/.active_config" ]; then
      warn "re-running $tag: the previous result predates the current deployment"
    else
      log "skip $tag (healthy result from the current deployment; FORCE=true to rerun)"; return
    fi
  fi
  log "[$cfg] $pattern  ->  $ep"
  "${BX[@]}" python3 /scripts/scrape_metrics.py --out "$RESULTS/metrics_${tag}_before.txt" "${targets[@]}"

  local common=(--backend openai --base-url "$ep" --endpoint /v1/completions
                --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_ID" --seed 42
                --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99
                --save-result --result-dir "$RESULTS" --result-filename "results_${tag}.json")
  case "$pattern" in
    rate*)
      "${BX[@]}" env OPENAI_API_KEY="$key" vllm bench serve "${common[@]}" \
        --dataset-name random --random-input-len "$BENCH_INPUT_LEN" --random-output-len "$BENCH_OUTPUT_LEN" \
        --ignore-eos --num-prompts "$BENCH_NUM_PROMPTS" --request-rate "${pattern#rate}" \
        2>&1 | tee "$OUT/log_${tag}.txt" | { grep -vE '^\s*[0-9]+%|it/s' || true; } ;;
    multiturn)
      "${BX[@]}" python3 /scripts/multiturn_bench.py --base-url "$ep" --model "$SERVED_MODEL_NAME" \
        --api-key "$key" --sessions "$MT_SESSIONS" --turns "$MT_TURNS" --session-rate "$MT_SESSION_RATE" \
        --label "$tag" --result-file "$RESULTS/results_${tag}.json" 2>&1 | tee "$OUT/log_${tag}.txt" ;;
    sharegpt)
      "${BX[@]}" env OPENAI_API_KEY="$key" vllm bench serve "${common[@]}" \
        --dataset-name sharegpt --dataset-path /bench/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 250 --request-rate 4 2>&1 | tee "$OUT/log_${tag}.txt" | { grep -vE '^\s*[0-9]+%|it/s' || true; } ;;
  esac
  "${BX[@]}" python3 /scripts/scrape_metrics.py --out "$RESULTS/metrics_${tag}_after.txt" "${targets[@]}"
  if [ "$(result_label "$cfg")" = "C" ]; then   # Mooncake evidence for the ablation table
    kubectl -n kv-tier logs deploy/mooncake-master --since=30m > "$OUT/mooncake_master_${tag}.log" 2>&1 || true
  fi
  sleep 10   # let queues drain between runs
}

for cfg in $CONFIGS; do
  T_CFG=$SECONDS
  if [ "$(cat "$BP1_ROOT/.active_config" 2>/dev/null)" != "$cfg" ] || [ "${FORCE_REDEPLOY:-false}" = "true" ]; then
    set_config "$cfg"
  fi
  ep="$(endpoint_for "$cfg")"; key=""
  if [ "${BENCH_VIA_GATEWAY:-false}" = "true" ] && [ "$cfg" != "A" ]; then
    ensure_gateway_key; ep="http://litellm.inference.svc.cluster.local:4000"; key="$LITELLM_MASTER_KEY"
  fi
  read -r -a targets <<< "$(metric_targets "$cfg")"

  auth=(); [ -n "$key" ] && auth=(-H "Authorization: Bearer $key")
  log "[$cfg] warm-up"
  for _ in 1 2 3; do
    "${BX[@]}" curl -sf "$ep/v1/completions" -H 'Content-Type: application/json' "${auth[@]}" \
      -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"warm up\",\"max_tokens\":16}" >/dev/null \
      || die "warm-up request to $ep failed"
  done

  for pattern in $PATTERNS; do
    T_PAT=$SECONDS
    run_pattern "$cfg" "$pattern" "$ep" "$key" "${targets[@]}"
    log "[$cfg] $pattern took $(( SECONDS - T_PAT ))s"
  done
  log "[$cfg] done in $(( (SECONDS - T_CFG) / 60 ))m$(( (SECONDS - T_CFG) % 60 ))s (incl. any redeploy)"
done

log "Building the results table"
"${BX[@]}" python3 /scripts/analyze.py "$RESULTS"
pass "$PROFILE run of [$CONFIGS] done in $(( (SECONDS - T_START) / 60 ))m — results in $OUT (summary.md / summary.csv)"
