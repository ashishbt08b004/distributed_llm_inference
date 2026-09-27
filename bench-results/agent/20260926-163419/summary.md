# Analyst crew run — 2026-09-26 16:35 — C no-kv-tier, one-call-per-reply

- endpoint: `hosted_vllm/qwen7b` @ `http://localhost:4000/v1`
- runs: 12 (concurrency 8, repeat 1), elapsed 68 s
- **accuracy: 5/12**
- tool calls: total 322, per run mean 26.8 / max 38
- LLM calls: total 364, per run mean 30.3
- tokens in/out: 568560 / 20534  (8314 prompt tok/s offered)
- wall per run: p50 32.43 s, p95 58.56 s
- sampling: temperature 0.3, top_p 0.8, top_k 20, repetition_penalty 1.05; max_iter 15, guardrail retries 2

**Why runs failed:** error 1 (the run crashed or an LLM call failed), wrong_value 6 (ran queries and answered, but the value is wrong)

| run | difficulty | correct | expected | got | failure | tool calls | LLM calls | guardrail retries | wall s |
|---|---|---|---|---|---|---|---|---|---|
| q01 r0 | easy | ✅ | 103 | 103 |  | 19 | 22 | 0 | 13.79 |
| q02 r0 | medium | ✅ | 541318.93 | 541318.93 |  | 36 | 41 | 2 | 43.05 |
| q03 r0 | medium | ❌ | Electronics | Toys | wrong_value | 25 | 28 | 0 | 32.43 |
| q04 r0 | medium | ❌ | East | store | wrong_value | 38 | 43 | 2 | 55.32 |
| q05 r0 | medium | ❌ | 12.9 | 100.0 | wrong_value | 20 | 24 | 1 | 53.44 |
| q06 r0 | hard | ✅ | BEAU-03 | BEAU-03 |  | 31 | 34 | 0 | 58.56 |
| q07 r0 | hard | ❌ | 15.85 | 1.06 | wrong_value | 24 | 27 | 0 | 23.75 |
| q08 r0 | hard | ❌ | 9.5 | 0.0 | wrong_value | 21 | 24 | 0 | 20.51 |
| q09 r0 | hard | ❌ | South | Exception: Task failed guardrail validation after 2 retries. Last error: Your answer contains a tool call written as text (JSON or <tool_call> tags). Text is not executed. Call the tool through the tool-calling interface, wait for its result, and only then write your answer in plain prose. | error | 20 | 23 | 3 | 27.7 |
| q10 r0 | hard | ❌ | 54.9 | 60.4 | wrong_value | 37 | 41 | 1 | 43.61 |
| q11 r0 | hard | ✅ | 144 | 144 |  | 25 | 28 | 0 | 29.58 |
| q12 r0 | hard | ✅ | 2.79 | 2.78 |  | 26 | 29 | 0 | 28.03 |

## Cluster metrics over the run (Prometheus)

| metric | value |
|---|---|
| router_queue_depth_max | 0.0 |
| router_rejected | None |
| router_queue_wait_avg_s | None |
| vllm_prefix_hit_rate | 0.9307 |
| vllm_ttft_avg_s | 0.0796 |
| vllm_waiting_max | 0.0 |
| vllm_preemptions | 0.0 |
