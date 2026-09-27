# Blueprint 2 — Multi-GPU LLM inference stack on 1× 8×A100 node

**Goal:** rerun the same architecture from Blueprint 1, but on a Lambda **8× A100 SXM 80GB** node so that (a) each pod gets its own physical GPU with real NVLink between them, (b) we can serve a larger model with tensor parallelism, and (c) the prefill/decode disaggregation shows measurable latency wins rather than being architecturally-real-but-numerically-noisy.

**Target model:** `meta-llama/Llama-3.1-70B-Instruct` (accept the license first at HuggingFace) or `Qwen/Qwen2.5-72B-Instruct` (no license gating).

**Budget target:** ~$100-120 (~8 hours of 8× A100 SXM at ~$14/hr).

**Assumes you completed Blueprint 1** and have working knowledge of the manifests, benchmark scripts, and result-analysis workflow. This blueprint highlights only what differs.

---

## What's different from Blueprint 1

| Aspect | Blueprint 1 (H100 single) | Blueprint 2 (8× A100) |
|---|---|---|
| Cluster | k3s, 1 node, 1 GPU | k3s, 1 node, 8 GPUs |
| GPU allocation | HAMi fractional slicing | Whole-GPU per pod (no HAMi) |
| Model | Qwen2.5-7B (fits easily) | Qwen2.5-72B or Llama-70B, tensor-parallel |
| Prefill pods | 2 × (0.25 GPU each) | 2 pods × TP=2 = 4 GPUs total |
| Decode pods | 3 × (0.15 GPU each) | 2 pods × TP=2 = 4 GPUs total |
| Interconnect | Memory bus only | NVLink at ~600 GB/s peer-to-peer |
| Mooncake tier | CPU RAM + one NVMe | CPU RAM only (Lambda 8×A100 has enough RAM to skip NVMe) |
| Router | SGLang router | SGLang router (unchanged) |
| Gateway | LiteLLM | LiteLLM (unchanged) |
| Expected disagg win | Marginal | 2-4× on prefill-heavy bursts |

**Why no HAMi:** each vLLM pod needs a whole GPU for tensor parallelism to work correctly — NCCL requires exclusive access to the CUDA context. HAMi's fractional model isn't needed here because 8 GPUs ÷ 4 pods = 2 GPUs per pod, exactly what TP=2 wants.

---

## Phase 0 — Provision, cost discipline

**Time:** 15 min. **Cost so far:** $0.

**Budget rule:** never leave the 8× A100 instance idle. Every hour is $14. The strategy is to prepare *everything* — manifests, scripts, test data — in a separate cheap workspace before launching. When you launch, execute the run in a focused 6-8 hour window and terminate immediately after.

Two ways to prepare:

**Option A (recommended):** keep the H100 node from Blueprint 1 alive for a few extra hours ($2.50/hr) and use it as a scratch workspace. Copy manifests over via SSH, edit them, test the YAML syntax with `kubectl apply --dry-run=client`. Then terminate the H100 and spin up the 8× A100.

**Option B:** prepare everything locally on your laptop, tested by dry-run only. Less certainty, cheaper.

When ready:

```bash
# Launch 8× A100 SXM 80GB via Lambda console
# Note the new public IP
export LAMBDA_IP=<new-ip>
ssh -i $SSH_KEY ubuntu@$LAMBDA_IP

# Verify
nvidia-smi   # 8 GPUs, 80 GB each
nvidia-smi topo -m   # verify NVLink between all pairs (should show NV18 or similar)
```

**Pass gate:** `nvidia-smi topo -m` shows NVLink connectivity between all 8 GPU pairs (not just PCIe).

---

## Phase 1 — Install k3s + NVIDIA GPU Operator (same as before, faster)

**Time:** 15 min. **Cost:** ~$4.

Exact same commands as Blueprint 1 Phase 1. K3s bootstraps, GPU Operator installs, node reports 8 GPUs:

```bash
kubectl describe node | grep -A2 Capacity | grep nvidia
# Should show: nvidia.com/gpu: 8
```

**Skip HAMi entirely.** We're using whole GPUs.

---

## Phase 2 — Observability stack

**Time:** 10 min. **Cost:** ~$6.

Exact same install as Blueprint 1 Phase 3:

```bash
helm install monitor prometheus-community/kube-prometheus-stack \
    -n observability --create-namespace \
    --set grafana.adminPassword=admin \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --wait
```

Port-forward Grafana as before. Confirm all 8 GPUs are reporting.

---

## Phase 3 — Deploy Mooncake KV storage tier

**Time:** 15 min. **Cost:** ~$8.

Similar to Blueprint 1 but scale the storage tier up to match the larger model — 72B produces bigger KV blocks.

Key changes from the single-node manifest:

```yaml
# In the mooncake-store Deployment spec:
resources:
  requests: { cpu: "4", memory: "80Gi" }   # bigger CPU RAM tier
  limits:   { cpu: "8", memory: "100Gi" }
```

Skip the NVMe PV entirely — 100 GB of CPU RAM as the pool tier is plenty for a benchmark. Lambda 8× A100 nodes typically have 1 TB+ of RAM.

If Mooncake integration proved unreliable during Blueprint 1, **run Blueprint 2 without it** for the first pass. You'll still see the value of tensor parallelism, prefix caching, and cache-aware routing. Add Mooncake back only if the first pass has time to spare.

---

## Phase 4 — Deploy prefill and decode pools with tensor parallelism

**Time:** 30 min (includes model download, which is large). **Cost:** ~$15.

This is where the manifests substantively change. Two prefill pods and two decode pods, each requesting 2 whole GPUs for TP=2.

**Pod-to-GPU placement:** by default, K8s will assign whichever 2 GPUs are free. For NVLink locality on 8× A100, GPU pairs (0,1), (2,3), (4,5), (6,7) are usually adjacent — but Lambda's exact NVLink topology varies. Check with `nvidia-smi topo -m` and if you see any pair with less than NV18 (18-lane NVLink), consider setting `NVIDIA_VISIBLE_DEVICES` explicitly.

For simplicity, let K8s handle placement — it's fine for a benchmark.

### Pre-download the model

The 72B model is ~150 GB. Download once to hostPath, all pods reuse it:

```bash
# On node
export HF_TOKEN=<your-token>
docker run --rm -e HF_TOKEN \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.9.0 \
    huggingface-cli download Qwen/Qwen2.5-72B-Instruct
```

This takes 15-20 minutes on Lambda's fast network. Do it once before scaling up pods.

### Prefill pool (2 pods × TP=2 = 4 GPUs)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prefill
  namespace: inference
spec:
  replicas: 2
  strategy:
    type: Recreate   # avoid two versions competing for GPUs during rollouts
  selector: { matchLabels: { app: vllm, role: prefill } }
  template:
    metadata:
      labels: { app: vllm, role: prefill }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.9.0
        args:
          - "--model=Qwen/Qwen2.5-72B-Instruct"
          - "--served-model-name=qwen72b"
          - "--host=0.0.0.0"
          - "--port=8000"
          - "--max-model-len=8192"
          - "--tensor-parallel-size=2"
          - "--gpu-memory-utilization=0.90"
          - "--enable-prefix-caching"
          - "--kv-transfer-config"
          - '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
        env:
        - name: HF_TOKEN
          valueFrom: { secretKeyRef: { name: hf-token, key: token } }
        - name: MOONCAKE_CONFIG_PATH
          value: /config/mooncake.json
        - name: NCCL_DEBUG
          value: WARN
        ports: [{ containerPort: 8000, name: http }]
        volumeMounts:
        - { name: config,   mountPath: /config }
        - { name: hf-cache, mountPath: /root/.cache/huggingface }
        - { name: shm,      mountPath: /dev/shm }
        resources:
          limits:
            nvidia.com/gpu: 2
      volumes:
      - name: config
        configMap: { name: mooncake-config }
      - name: hf-cache
        hostPath: { path: /home/ubuntu/hf-cache, type: Directory }
      - name: shm
        emptyDir: { medium: Memory, sizeLimit: 16Gi }
```

**Two important details:**

1. **The `shm` volume.** vLLM uses shared memory for inter-process communication between TP workers. Default K8s pod shm size is 64 MB, which is too small. Mount `emptyDir` with `medium: Memory` and 16 GB size. Skip this and TP will fail with cryptic NCCL errors.

2. **`--gpu-memory-utilization=0.90`** — higher than the single-node 0.75 because we no longer share GPUs. Each pod owns its 2 GPUs completely, so we can push the KV pool up.

### Decode pool (2 pods × TP=2 = 4 GPUs)

Same manifest as prefill, with these changes:
- `name: vllm-decode`
- `role: decode`
- `--kv-transfer-config: kv_role=kv_consumer`
- Optionally different max_model_len if you want to bias decode toward longer contexts

Total: 8 GPUs used, 4 by prefill pool + 4 by decode pool.

**Sanity check before applying:** `2 prefill pods × 2 GPUs each + 2 decode pods × 2 GPUs each = 8 GPUs`. If you also want a spare pod for warm-up or a third pool, you're out of GPUs — the node has 8 and every one is committed.

### Deploy and wait

```bash
kubectl apply -f prefill-deploy.yaml
kubectl apply -f decode-deploy.yaml

# Watch startup — first startup is slow because vLLM has to warm up CUDA graphs
kubectl -n inference get pods -w
```

Startup takes ~5 minutes per pod on TP=2 for the 72B model. If it stalls for more than 8 minutes, check logs:

```bash
kubectl -n inference logs -f deploy/vllm-prefill
# Look for "torch.compile takes X.XX s in total" and then "Application startup complete"
```

**Pass gate:**

```bash
kubectl -n inference get pods       # 2 prefill + 2 decode all 1/1 Ready
kubectl -n inference port-forward svc/vllm-decode 8001:8000 &
curl http://localhost:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen72b","prompt":"Explain tensor parallelism briefly:","max_tokens":100}'
```

The response should stream a coherent completion in ~2-3 seconds.

---

## Phase 5 — Router and gateway

**Time:** 5 min. **Cost:** ~$16.

Identical to Blueprint 1 Phases 6-7. Same manifests, same NodePort, same test. Only difference: change `qwen7b` to `qwen72b` in configs.

---

## Phase 6 — Benchmark grid

**Time:** ~4 hours. **Cost:** ~$56.

The grid is broadly the same as Blueprint 1, but with two changes:

**1. Larger request sizes.** 70B models are typically hit with longer prompts and longer generations. Use:
- `--random-input-len 1024`  (was 512)
- `--random-output-len 256`  (was 128)

**2. Add a fourth arrival pattern: `rate=32`** — a sustained heavy-load scenario that would have OOMed on Blueprint 1's setup but should work here.

Same three configs (A: baseline, B: prefix-cached, C: full disaggregation). Same benchmark commands, adjusted args:

```bash
for RATE in 1 8 32 inf; do
  vllm bench serve \
    --backend openai \
    --base-url http://localhost:8080 \
    --model qwen72b \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 400 \
    --request-rate $RATE \
    --save-result \
    --result-filename results_C_rate${RATE}.json
done

# Multi-turn — key for showing Mooncake / prefix cache value
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8080 \
  --model qwen72b \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 8 \
  --save-result \
  --result-filename results_C_sharegpt.json
```

### What to compare

The interesting numbers on 8× A100 that Blueprint 1 couldn't produce:

| Comparison | What it isolates |
|---|---|
| Config A (single pod, TP=2) vs Config B (2 decode pods behind router) | Value of horizontal scaling with cache-aware routing |
| Config B (prefix caching, no disagg) vs Config C (full disagg with Mooncake) | Value of disaggregation and cross-pod KV sharing |
| Random-token vs ShareGPT | Impact of workload realism on cache economics |
| rate=8 vs rate=32 vs rate=inf | The utilization wall — where Pollaczek-Khinchine starts biting |

Expected shape of results if everything works:

- **TTFT p95:** Config C at rate=inf should be ~1.5-2× better than Config A on ShareGPT (prefill workers stay busy, decode workers don't get held up prefilling).
- **Throughput at rate=inf:** Config C should push ~3000-5000 tok/s aggregate on Qwen-72B (vs ~1000-1500 for Config A).
- **Prefix cache hit rate on ShareGPT:** Config B ~40-60%, Config C ~70-85% (Mooncake extends the cache across pods).
- **Preemptions:** should be 0 across all runs at `gpu-memory-utilization=0.90`. If you see preemptions, drop to 0.85 and rerun.

---

## Phase 7 — Cost checkpoint and go/no-go

At end of Phase 6, total spend for Blueprint 2 should be ~$80. If you're running under budget, options:

1. **Repeat the ShareGPT run with different concurrency** to build a proper throughput-vs-latency curve
2. **Try a longer max_model_len (16k)** to see how KV growth affects the picture
3. **Attempt llm-d instead of SGLang router** (see Blueprint 1 addendum)
4. **Just teardown and save the money**

---

## Phase 8 — Teardown

**Time:** 10 min.

**This is the most important phase from a budget standpoint.** Even a couple of forgotten hours cost $28.

```bash
# Copy results back to laptop first
scp -i $SSH_KEY -r ubuntu@$LAMBDA_IP:~/bench ./results-multi-gpu

# Verify results are on your laptop before proceeding
ls -la ./results-multi-gpu

# THEN terminate the instance via Lambda console (not just stop)
# Confirm in the console that the instance status is "Terminated" and not "Stopped"
```

Set a calendar alarm for 30 minutes after you plan to be done, as a hard "check that the instance is terminated" reminder.

---

## Deliverables from multi-GPU phase

- 15 benchmark JSON files (3 configs × 5 arrival patterns) + 3 ShareGPT runs
- 15 metrics snapshots
- A results comparison across configs and workloads with real disaggregation numbers
- One clean, verified shutdown

Combined with Blueprint 1 results, you have a coherent story: single-node shows the architecture works and every layer is real; multi-GPU shows the layers actually produce measurable wins on production-shaped workloads.

---

## Differences from Blueprint 1 that trip people up

| Issue | Why it happens | Fix |
|---|---|---|
| NCCL init hangs on pod startup | `/dev/shm` too small for TP=2 workers | 16 GB emptyDir Memory volume mounted at `/dev/shm` |
| Second prefill pod never schedules | Only 4 GPUs left after first pod grabbed 2 | Verify GPU allocation math: 2×TP + 2×TP = 8, no spare |
| vLLM startup times out (10+ min) | Model download on cold pod | Pre-download model to hostPath once before scaling replicas |
| ImagePullBackOff on vllm image | Docker Hub rate limit on Lambda IPs | Use `docker login` on the node first, or mirror to a private registry |
| NCCL P2P warnings | Non-adjacent GPU pairs assigned | Usually harmless; if performance suffers, use `nodeSelector` or explicit `NVIDIA_VISIBLE_DEVICES` |
| Grafana shows only 1-2 GPUs | DCGM exporter not scraping all devices | Check `kubectl -n gpu-operator get pods` — all DCGM pods should be Running |
| Out of CPU RAM after Mooncake deploy | Storage pod requesting too much | Reduce Mooncake `cpu-mem-gb` argument to match available RAM |

---

## What each blueprint does and doesn't prove

**Blueprint 1 (single-H100) tells you:**
- Can I stand up all five layers?
- Do the components talk to each other?
- Does the operational surface work end-to-end?
- Directional: does prefix caching help?

**Blueprint 2 (8× A100) tells you:**
- Do the layers produce quantifiable wins?
- What's the actual disaggregation gain?
- How does TP=2 scale for a 70B model?
- Realistic TTFT/ITL numbers on production-shaped workloads

Neither one proves cross-node RDMA behavior (which needs a true multi-node cluster with InfiniBand — Lambda's 1-Click Clusters, out of budget). But everything else about the architecture is genuinely exercised.

---

## Final artifacts checklist

By the end of both blueprints, you should have:

- [ ] All manifests archived in git
- [ ] Benchmark result JSON from both runs
- [ ] Grafana dashboard exports (JSON) from both runs
- [ ] A comparison table: single-node vs multi-GPU on the same metrics
- [ ] Total Lambda spend under $250 (leaving buffer for the ~$150 you haven't used)
- [ ] Both instances fully terminated in the Lambda console

That's a complete production LLM inference architecture demonstrated, benchmarked, and torn down on a graduate-student budget.
