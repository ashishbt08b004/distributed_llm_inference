# Analyst crew run — 2026-09-26 16:19 — C no-kv-tier, guardrails

- endpoint: `hosted_vllm/qwen7b` @ `http://localhost:4000/v1`
- runs: 12 (concurrency 4, repeat 1), elapsed 215 s
- **accuracy: 3/12**
- tool calls: total 366, per run mean 30.5 / max 93
- LLM calls: total 214, per run mean 17.8
- tokens in/out: 427681 / 46233  (1985 prompt tok/s offered)
- wall per run: p50 62.42 s, p95 113.18 s
- sampling: temperature 0.3, top_p 0.8, top_k 20, repetition_penalty 1.05; max_iter 15, guardrail retries 2

**Why runs failed:** error 2 (the run crashed or an LLM call failed), wrong_value 7 (ran queries and answered, but the value is wrong)

| run | difficulty | correct | expected | got | failure | tool calls | LLM calls | guardrail retries | wall s |
|---|---|---|---|---|---|---|---|---|---|
| q01 r0 | easy | ❌ | 103 | 102 | wrong_value | 11 | 14 | 1 | 34.67 |
| q02 r0 | medium | ❌ | 541318.93 | 653054.61 | wrong_value | 9 | 12 | 0 | 23.49 |
| q03 r0 | medium | ✅ | Electronics | Electronics |  | 21 | 16 | 0 | 42.32 |
| q04 r0 | medium | ❌ | East | Exception: Task failed guardrail validation after 2 retries. Last error: Your answer contains a tool call written as text (JSON or <tool_call> tags). Text is not executed. Call the tool through the tool-calling interface, wait for its result, and only then write your answer in plain prose. | error | 17 | 9 | 3 | 76.25 |
| q05 r0 | medium | ✅ | 12.9 | 12.9 |  | 93 | 16 | 0 | 84.24 |
| q06 r0 | hard | ❌ | BEAU-03 | Widget X | wrong_value | 25 | 30 | 1 | 113.18 |
| q07 r0 | hard | ✅ | 15.85 | 15.85 |  | 23 | 11 | 1 | 55.24 |
| q08 r0 | hard | ❌ | 9.5 | 0.0 | wrong_value | 17 | 20 | 0 | 55.67 |
| q09 r0 | hard | ❌ | South | North | wrong_value | 28 | 20 | 0 | 85.39 |
| q10 r0 | hard | ❌ | 54.9 | 61.5 | wrong_value | 70 | 35 | 0 | 102.67 |
| q11 r0 | hard | ❌ | 144 | Exception: Task failed guardrail validation after 2 retries. Last error: Your answer contains a tool call written as text (JSON or <tool_call> tags). Text is not executed. Call the tool through the tool-calling interface, wait for its result, and only then write your answer in plain prose. | error | 23 | 16 | 3 | 62.42 |
| q12 r0 | hard | ❌ | 2.79 | NULL | wrong_value | 29 | 15 | 0 | 44.09 |

## Cluster metrics over the run (Prometheus)

| metric | value |
|---|---|
| router_queue_depth_max | 0.0 |
| router_rejected | None |
| router_queue_wait_avg_s | None |
| vllm_prefix_hit_rate | 0.8475 |
| vllm_ttft_avg_s | 0.0838 |
| vllm_waiting_max | 0.0 |
| vllm_preemptions | 0.0 |
