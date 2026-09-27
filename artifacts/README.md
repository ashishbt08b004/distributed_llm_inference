# Artifacts

Diagrams, screenshots, charts and documents for the project, all in formats GitHub renders in the browser. Everything below displays inline; click any image for full size.

## Conventions (for new artifacts)

| Kind | Folder | Format | Naming |
|---|---|---|---|
| Architecture / flow diagrams | `diagrams/` | SVG (or PNG) | `<subject>-architecture.svg` |
| Screenshots (Grafana, terminals, UIs) | `screenshots/<source>/` | PNG | `YYYY-MM-DD_<what-it-shows>.png` |
| Charts made from results | `charts/` | PNG (or SVG) | `<metric>-<breakdown>.png` |
| Documents and write-ups | `docs/` | PDF | `<topic>.pdf` |

- **No Office files.** `.docx`, `.pptx` and `.xlsx` don't render on GitHub and are git-ignored. Export documents to PDF and screenshots to PNG. The original can stay on disk next to the export; git ignores it.
- **Use lowercase kebab-case names** without spaces, so links and URLs stay clean.
- **Add every new artifact to the gallery below**, with a one-line caption saying what it shows and which run it's from.
- **Raw results don't belong here.** CSV, JSON and logs go in [`../bench-results/`](../bench-results/); this folder is for things people look at.

## Diagrams

**Blueprint 1: single-node stack on 1× H100.** The router Service fronts `kv_router` (admission queue, prefix routing, prefill→decode hop; configuration C) or `sglang_router` (configuration B). The pod boxes show the original plan's slice sizes; as deployed, prefill slices are 16 GB and decode slices 14 GB.

![Blueprint 1 architecture](diagrams/blueprint1-architecture.svg)

**Agent system: the analyst crew.** The laptop-side CrewAI crew, its 9 local tools, the grader and the report, connected through the SSH tunnel to the gateway, router and vLLM pods.

![Analyst crew architecture](diagrams/analyst-crew-architecture.svg)

**Blueprint 2: multi-GPU stack on 8× A100** (planned).

![Blueprint 2 architecture](diagrams/blueprint2-architecture.svg)

## Grafana screenshots, 26 Sep 2026 (16:22–16:52)

Dashboard "Blueprint 1 — Inference stack", configuration C with the Mooncake KV tier off. Three bursts are visible: the agent evaluation at concurrency 4 (≈ 16:28–16:30), at concurrency 8 (≈ 16:34–16:35), and the 400-request `rateinf` burst (≈ 16:40–16:41).

**Latency and throughput per pod.** The burst drives TTFT p95 to ≈ 6 s and inter-token p95 to ≈ 280 ms; the agent runs stay sub-second.

![Latency and throughput](screenshots/grafana/2026-09-26_latency-and-throughput.png)

**KV cache and scheduling.** Prefix-cache hit rate holds at 85–95% during the agent runs. At the burst, running requests reach ≈ 60 per pod while vLLM's waiting count stays near 0; KV usage and preemptions peak on `vllm-decode-2`, the pod with the smallest KV cache.

![KV cache and scheduling](screenshots/grafana/2026-09-26_kv-cache-and-scheduling.png)

**Router and GPU.** Requests spread across the three decode pods; `kv_router`'s prefix-match ratio is ≈ 90% for agent traffic. The GPU panel is empty because DCGM metrics were not being collected.

![Router and GPU](screenshots/grafana/2026-09-26_router-and-gpu.png)

**Admission queue.** Only the 400-request burst queues (peak ≈ 210); the agent runs never do. The queue-wait panel misses the burst because the router's wait counters appear only on first use.

![Admission queue](screenshots/grafana/2026-09-26_admission-queue.png)

## Charts

**Agent evaluation outcomes by run** (12 questions each), from [`../bench-results/agent/agent_runs.csv`](../bench-results/agent/agent_runs.csv). The fixes between runs removed the mechanical failures; the remaining misses are mostly wrong values.

![Agent run outcomes](charts/agent-run-outcomes.png)

**Multi-turn benchmark: time to first token by turn**, from [`../bench-results/quick/results_C_multiturn.json`](../bench-results/quick/results_C_multiturn.json). Later turns are faster despite longer prompts, because of prefix caching (85.6% hit rate).

![Multi-turn TTFT by turn](charts/multiturn-ttft-by-turn.png)

## Documents

- [Blueprint 1: Kubernetes manifests explained (PDF)](docs/blueprint1-kubernetes-manifests-explained.pdf)
