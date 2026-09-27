# Analyst crew run — 2026-09-26 16:30 — C no-kv-tier, one-call-per-reply

- endpoint: `hosted_vllm/qwen7b` @ `http://localhost:4000/v1`
- runs: 12 (concurrency 4, repeat 1), elapsed 132 s
- **accuracy: 6/12**
- tool calls: total 348, per run mean 29.0 / max 46
- LLM calls: total 388, per run mean 32.3
- tokens in/out: 635819 / 24839  (4804 prompt tok/s offered)
- wall per run: p50 32.93 s, p95 72.92 s
- sampling: temperature 0.3, top_p 0.8, top_k 20, repetition_penalty 1.05; max_iter 15, guardrail retries 2

**Why runs failed:** error 2 (the run crashed or an LLM call failed), wrong_value 4 (ran queries and answered, but the value is wrong)

| run | difficulty | correct | expected | got | failure | tool calls | LLM calls | guardrail retries | wall s |
|---|---|---|---|---|---|---|---|---|---|
| q01 r0 | easy | ✅ | 103 | 103 |  | 19 | 22 | 0 | 15.92 |
| q02 r0 | medium | ✅ | 541318.93 | 541318.93 |  | 42 | 47 | 2 | 39.35 |
| q03 r0 | medium | ✅ | Electronics | Electronics |  | 25 | 28 | 0 | 28.88 |
| q04 r0 | medium | ❌ | East | Central | wrong_value | 26 | 29 | 0 | 34.4 |
| q05 r0 | medium | ❌ | 12.9 | 28.7 | wrong_value | 28 | 31 | 0 | 38.21 |
| q06 r0 | hard | ❌ | BEAU-03 | Exception: Task failed guardrail validation after 2 retries. Last error: That is not an answer from the data. The data exists: call list_tables and describe_table to find the real table and column names, run the query, and answer with the value it returns. Never assume or invent results. | error | 46 | 51 | 3 | 65.26 |
| q07 r0 | hard | ✅ | 15.85 | 15.85 |  | 19 | 22 | 0 | 26.36 |
| q08 r0 | hard | ❌ | 9.5 | Exception: Task failed guardrail validation after 2 retries. Last error: Your answer contains a tool call written as text (JSON or <tool_call> tags). Text is not executed. Call the tool through the tool-calling interface, wait for its result, and only then write your answer in plain prose. | error | 20 | 23 | 3 | 22.66 |
| q09 r0 | hard | ❌ | South | East | wrong_value | 40 | 43 | 0 | 72.92 |
| q10 r0 | hard | ❌ | 54.9 | 60.4 | wrong_value | 27 | 30 | 0 | 32.93 |
| q11 r0 | hard | ✅ | 144 | 144 |  | 23 | 26 | 0 | 25.37 |
| q12 r0 | hard | ✅ | 2.79 | 2.78 |  | 33 | 36 | 0 | 32.79 |

## Cluster metrics over the run (Prometheus)

| metric | value |
|---|---|
| router_queue_depth_max | 0.0 |
| router_rejected | None |
| router_queue_wait_avg_s | None |
| vllm_prefix_hit_rate | 0.8859 |
| vllm_ttft_avg_s | 0.0745 |
| vllm_waiting_max | 0.0 |
| vllm_preemptions | 0.0 |
