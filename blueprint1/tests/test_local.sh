#!/usr/bin/env bash
# Local (no GPU) test of kv_router, multiturn_bench, scrape_metrics and analyze
# against fake vLLM servers. Needs python3 + aiohttp.   ./tests/test_local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT

for p in 18101 18102 18201 18202 18203; do
  python3 "$ROOT/tests/fake_vllm.py" "$p" & PIDS+=($!)
done
PREFILL_URLS="http://127.0.0.1:18101,http://127.0.0.1:18102" \
DECODE_URLS="http://127.0.0.1:18201 http://127.0.0.1:18202 http://127.0.0.1:18203" \
PORT=18080 METRICS_PORT=18090 LOG_LEVEL=WARNING python3 "$ROOT/router/kv_router.py" & PIDS+=($!)
for _ in $(seq 50); do curl -sf localhost:18080/health >/dev/null 2>&1 && break; sleep 0.2; done

echo "--- streaming completion through P/D router"
OUT=$(curl -sfN localhost:18080/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen7b","prompt":"hello world","max_tokens":5,"stream":true}')
echo "$OUT" | grep -q '\[DONE\]' || { echo "FAIL: no [DONE] in stream"; exit 1; }
PF=$(curl -s localhost:18101/_seen; curl -s localhost:18102/_seen)
echo "$PF" | grep -q '"max_tokens": 1' || { echo "FAIL: prefill did not get max_tokens=1: $PF"; exit 1; }
echo "ok: prefill hop got max_tokens=1, decode streamed"

echo "--- stickiness: same long prefix -> same decode pod"
LONG=$(python3 -c "print('shared system prompt ' * 100)")
for i in 1 2 3 4 5 6; do
  curl -sf localhost:18080/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen7b\",\"prompt\":\"$LONG q$i\",\"max_tokens\":2}" >/dev/null
done
HITS=$(for p in 18201 18202 18203; do curl -s localhost:$p/_seen | python3 -c "import sys,json;print(len(json.load(sys.stdin)))"; done | sort -n | tail -1)
[ "$HITS" -ge 6 ] || { echo "FAIL: long-prefix requests were spread out (max per pod $HITS)"; exit 1; }
echo "ok: all 6 shared-prefix requests went to one decode pod"
curl -s localhost:18090/metrics | grep -E '^kv_router_prefix_(matched|query)'

echo "--- admission queue: 1 slot per decode pod, 2 queue places, slow tokens"
for p in 18301 18302; do python3 "$ROOT/tests/fake_vllm.py" "$p" 0.05 & PIDS+=($!); done
DECODE_URLS="http://127.0.0.1:18301 http://127.0.0.1:18302" MAX_INFLIGHT_PER_WORKER=1 QUEUE_MAX=2 \
PORT=18081 METRICS_PORT=18091 LOG_LEVEL=WARNING python3 "$ROOT/router/kv_router.py" & PIDS+=($!)
for _ in $(seq 50); do curl -sf localhost:18081/health >/dev/null 2>&1 && break; sleep 0.2; done
for i in 1 2 3 4 5 6; do
  curl -s -o /dev/null -w '%{http_code}\n' localhost:18081/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen7b\",\"prompt\":\"q$i\",\"max_tokens\":20,\"stream\":true}" > "$TMP/code$i" &
done
wait_codes() { for i in 1 2 3 4 5 6; do while [ ! -s "$TMP/code$i" ]; do sleep 0.1; done; done; }
wait_codes
OK=$(cat "$TMP"/code? | grep -c '^200$' || true); REJ=$(cat "$TMP"/code? | grep -c '^429$' || true)
[ "$OK" -eq 4 ] && [ "$REJ" -eq 2 ] || { echo "FAIL: expected 4x200 + 2x429, got: $(cat "$TMP"/code? | tr '\n' ' ')"; exit 1; }
MAXP=$(for p in 18301 18302; do curl -s localhost:$p/_seen | python3 -c "import sys,json;print(len(json.load(sys.stdin)))"; done | sort -n | tail -1)
[ "$MAXP" -le 3 ] || { echo "FAIL: one pod took $MAXP of 4 admitted requests"; exit 1; }
curl -s localhost:18091/metrics | grep -E '^kv_router_(queue_depth|rejected_total)'
echo "ok: 2 in flight, 2 queued, 2 rejected with 429"

echo "--- multi-turn bench + metrics + analyze"
mkdir -p "$TMP/r"
TARGETS="http://127.0.0.1:18201 http://127.0.0.1:18202 http://127.0.0.1:18203"
python3 "$ROOT/bench/scrape_metrics.py" --out "$TMP/r/metrics_C_multiturn_before.txt" $TARGETS
python3 "$ROOT/bench/multiturn_bench.py" --base-url http://127.0.0.1:18080 --sessions 8 --turns 3 \
  --session-rate 20 --max-tokens 16 --result-file "$TMP/r/results_C_multiturn.json"
python3 "$ROOT/bench/scrape_metrics.py" --out "$TMP/r/metrics_C_multiturn_after.txt" $TARGETS
# fabricate a config-B row to exercise the comparison table
python3 - "$TMP/r" <<'EOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); r = json.loads((d / "results_C_multiturn.json").read_text())
r["p95_ttft_ms"] *= 1.5; r["output_throughput"] *= 0.8
(d / "results_B_multiturn.json").write_text(json.dumps(r))
EOF
python3 "$ROOT/bench/analyze.py" "$TMP/r" | head -20
grep -q "| multiturn | C |" "$TMP/r/summary.md" || { echo "FAIL: summary missing row"; exit 1; }
echo "ALL LOCAL TESTS PASSED"
