# Analyst crew run — 2026-09-26 16:03

- endpoint: `hosted_vllm/qwen7b` @ `http://localhost:4000/v1`
- runs: 12 (concurrency 4, repeat 1), elapsed 118 s
- **accuracy: 1/12**
- tool calls: total 205, per run mean 17.1 / max 29
- LLM calls: total 107, per run mean 8.9
- tokens in/out: 209913 / 38144  (1785 prompt tok/s offered)
- wall per run: p50 37.94 s, p95 49.16 s

| run | difficulty | correct | expected | got | tool calls | LLM calls | wall s |
|---|---|---|---|---|---|---|---|
| q01 r0 | easy | ✅ | 103 | 103 | 13 | 7 | 13.12 |
| q02 r0 | medium | ❌ | 541318.93 |  | 13 | 19 | 34.94 |
| q03 r0 | medium | ❌ | Electronics |  | 14 | 6 | 37.94 |
| q04 r0 | medium | ❌ | East | 895.55 | 29 | 12 | 48.23 |
| q05 r0 | medium | ❌ | 12.9 |  | 26 | 9 | 49.16 |
| q06 r0 | hard | ❌ | BEAU-03 |  | 14 | 9 | 46.12 |
| q07 r0 | hard | ❌ | 15.85 |  | 17 | 17 | 39.15 |
| q08 r0 | hard | ❌ | 9.5 |  | 23 | 6 | 32.84 |
| q09 r0 | hard | ❌ | South |  | 8 | 7 | 38.01 |
| q10 r0 | hard | ❌ | 54.9 | 28.0"} | 19 | 5 | 29.52 |
| q11 r0 | hard | ❌ | 144 | 3 | 19 | 6 | 31.59 |
| q12 r0 | hard | ❌ | 2.79 | 78.54 | 10 | 4 | 23.78 |

## Cluster metrics over the run (Prometheus)

| metric | value |
|---|---|
| router_queue_depth_max | 0.0 |
| router_rejected | None |
| router_queue_wait_avg_s | None |
| vllm_prefix_hit_rate | 0.7775 |
| vllm_ttft_avg_s | 0.0819 |
| vllm_waiting_max | 0.0 |
| vllm_preemptions | 0.0 |
