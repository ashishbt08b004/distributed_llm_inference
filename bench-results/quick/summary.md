# Blueprint 1 — ablation results

| Pattern | Config | OK | TTFT p50 (ms) | TTFT p95 (ms) | ITL p95 (ms) | E2E p95 (ms) | Out tok/s | Prefix hit (HBM) | Connector hit (LMCache) | Preemptions | Mooncake ops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rate4 | C-nokv | 100 | 222 | 273 | 32.0 | 3,221 | 464 | 0.0% | n/a | 0 | 0 |
| rate4 | C | 100 | 254 | 381 | 28.3 | 3,195 | 465 | 0.0% | 15.5% | 0 | 1,107 |
| rateinf | C-nokv | 400 | 9,504 | 15,631 | 197.1 | 19,332 | 2,424 | 0.1% | n/a | 2 | 0 |
| rateinf | C | 400 | 8,687 | 16,779 | 40.1 | 19,258 | 2,646 | 23.8% | 32.8% | 0 | 3,124 |
| multiturn | C-nokv | 100 | 208 | 311 | 27.1 | 3,168 | 631 | 85.6% | n/a | 0 | 0 |
| multiturn | C | 100 | 285 | 492 | 27.4 | 3,278 | 622 | 85.2% | 15.8% | 0 | 548 |

## Layer contribution (relative change)

| Pattern | Step | TTFT p95 | ITL p95 | Out tok/s |
|---|---|---|---|---|
| rate4 | C-nokv → C | 39% | -12% | 0% |
| rateinf | C-nokv → C | 7% | -80% | 9% |
| multiturn | C-nokv → C | 58% | 1% | -1% |

## Multi-turn: mean TTFT by turn (ms)

| Config | turn 1 | turn 2 | turn 3 | turn 4 | turn 5 |
|---|---|---|---|---|---|
| C-nokv | 278 (~1,443 tok) | 191 (~1,609 tok) | 207 (~1,774 tok) | 207 (~1,940 tok) | 215 (~2,106 tok) |
| C | 373 (~1,443 tok) | 296 (~1,609 tok) | 270 (~1,774 tok) | 293 (~1,940 tok) | 278 (~2,105 tok) |
