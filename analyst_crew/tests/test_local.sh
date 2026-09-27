#!/usr/bin/env bash
# Offline test (no model, no cluster): data + tools + grading, then the full crew
# loop and the concurrent eval runner against a scripted fake LLM.
#   ./tests/test_local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
PORT=18999
cd "$ROOT"

echo "--- data + tools"
"$PY" make_data.py >/dev/null
"$PY" - <<'EOF'
import json, tools, run
q = {x["id"]: x for x in json.load(open("data/questions.json"))}
assert "customers: 500 rows" in tools.list_tables.run()
assert "ERROR" in tools.run_sql.run(query="DELETE FROM customers")
assert "ERROR" in tools.run_sql.run(query="SELECT 1; DROP TABLE customers")
assert tools.calculator.run(expression="round(10/4, 1)") == "2.5"
assert "ERROR" in tools.calculator.run(expression="__import__('os')")
assert "median=" in tools.column_stats.run(query="SELECT satisfaction FROM support_tickets")
assert run.grade(q["q01"], "103") and not run.grade(q["q01"], "104")
assert run.grade(q["q03"], "Electronics") and not run.grade(q["q03"], "Home")
assert run.grade(q["q02"], "$541,318.93") and run.extract_final("x\nFINAL_ANSWER: **42**") == "42"
print("ok: tools are read-only and safe, grader works")
EOF

echo "--- crew loop against a scripted fake LLM"
python3 tests/fake_llm.py $PORT & FAKE=$!
trap 'kill $FAKE 2>/dev/null || true' EXIT
sleep 0.5
export LLM_MODEL=hosted_vllm/qwen7b LLM_BASE_URL=http://127.0.0.1:$PORT/v1 LLM_API_KEY=dummy
export BENCH_RESULTS_DIR="$(mktemp -d)"   # keep test runs out of the real bench-results/
OUT=$("$PY" run.py ask --qid q01 2>&1)
echo "$OUT" | tail -3
echo "$OUT" | grep -q "correct: True" || { echo "FAIL: q01 not answered correctly"; exit 1; }
echo "$OUT" | grep -q "tool calls: 8 " || { echo "FAIL: expected 8 tool calls"; exit 1; }

echo "--- guardrail: a tool call written as text is rejected and the agent retries"
python3 tests/fake_llm.py $((PORT + 1)) --misbehave & FAKE2=$!
trap 'kill $FAKE $FAKE2 2>/dev/null || true' EXIT
sleep 0.5
OUT=$(LLM_BASE_URL=http://127.0.0.1:$((PORT + 1))/v1 "$PY" run.py ask --qid q01 2>&1)
echo "$OUT" | grep -E "correct:|guardrail"
echo "$OUT" | grep -q "correct: True" || { echo "FAIL: guardrail did not recover the run"; echo "$OUT" | tail -20; exit 1; }
echo "$OUT" | grep -q "analysis:text_tool_call" || { echo "FAIL: guardrail rejection not recorded"; exit 1; }

echo "--- concurrent eval runner + report CSVs"
EVAL=$("$PY" run.py eval --qids q01 --repeat 3 --concurrency 3 --label "offline test" 2>&1)
echo "$EVAL" | grep -E "accuracy|tool calls:"
echo "$EVAL" | grep -q "accuracy: 3/3" || { echo "FAIL: eval accuracy"; echo "$EVAL"; exit 1; }
RUN=$(ls -d "$BENCH_RESULTS_DIR"/agent/*/)
for f in summary.md meta.json results.jsonl q01_r0.trace.jsonl; do
  [ -s "$RUN/$f" ] || { echo "FAIL: $f missing"; exit 1; }
done
"$PY" - "$BENCH_RESULTS_DIR/agent" <<'PYEOF'
import csv, json, sys, pathlib
root = pathlib.Path(sys.argv[1])
runs = list(csv.DictReader(open(root / "agent_runs.csv")))
qs = list(csv.DictReader(open(root / "agent_questions.csv")))
assert len(runs) == 1 and runs[0]["label"] == "offline test" and runs[0]["accuracy"] == "1.0", runs
assert len(qs) == 3 and all(q["correct"] == "1" and q["tool_run_sql"] == "2" for q in qs), qs
trace = [json.loads(l) for l in open(next(root.glob("*/q01_r0.trace.jsonl")))]
assert len(trace) == 11 and {t["agent"] for t in trace} == {"Data Planner", "SQL Analyst", "Reviewer"}, trace
print(f"ok: agent_runs.csv (1 row), agent_questions.csv (3 rows), trace ({len(trace)} LLM calls)")
PYEOF
rm -rf "$BENCH_RESULTS_DIR"
echo "ALL LOCAL TESTS PASSED"
