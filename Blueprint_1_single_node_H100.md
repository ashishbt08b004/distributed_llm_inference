# Blueprint 1 — Single-node LLM inference stack on 1× H100

**Goal:** stand up every layer of the production inference architecture (gateway, cache-aware router, disaggregated prefill/decode pods, KV memory tiers with Mooncake, observability) on a **single Lambda H100 PCIe node**, then benchmark it with an ablation grid that isolates the contribution of each layer.

**Target model:** `Qwen/Qwen2.5-7B-Instruct` — small enough to run several partitioned pods on one 80 GB H100, large enough that prefill/decode split is meaningful, well-benchmarked.

**Budget target:** ~$100 (≈ 40 hours of 1× H100 PCIe at ~$2.50/hr). Stop-the-clock discipline is important — always destroy the instance when not actively working.

**Prerequisites (bring your own):**
- Lambda Cloud account with SSH key uploaded
- HuggingFace token with access to Qwen2.5-7B-Instruct (public model, any token works)
- Basic kubectl + Helm familiarity
- ~10 GB free on your laptop for kubeconfig, logs, benchmark results

**Architecture diagram:** [`artifacts/diagrams/blueprint1-architecture.png`](artifacts/diagrams/blueprint1-architecture.png). **Runnable implementation:** [`blueprint1/README.md`](blueprint1/README.md). It follows this plan, fixes the parts that don't work as written, and adds a second router (`kv_router`) with an admission queue (see Phase 6).

---

## Phase 0 — Provision node and connect

**Time:** 10 min. **Cost so far:** $0.

Launch a 1× H100 PCIe instance from Lambda Cloud console. Choose the Ubuntu 22.04 image with CUDA 12.4 preinstalled. Note the public IP.

From your laptop:

```bash
export LAMBDA_IP=<public-ip-from-console>
export SSH_KEY=~/.ssh/lambda_key
ssh -i $SSH_KEY ubuntu@$LAMBDA_IP

# On the node — verify GPU and driver
nvidia-smi   # should show 1× H100, driver 550+, CUDA 12.4+
```

**Pass gate:** `nvidia-smi` shows one H100 with 80 GB HBM and no processes running.

---

## Phase 1 — Install k3s + NVIDIA GPU support

**Time:** 20 min. **Cost so far:** ~$1.

K3s is a lightweight Kubernetes distribution — one binary, no etcd cluster, ideal for single-node work. It's a real conformant K8s, not a toy.

```bash
# Install k3s server, disable Traefik (we'll use our own gateway)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik --write-kubeconfig-mode=644" sh -

# Verify
sudo systemctl status k3s
kubectl get nodes   # should show one Ready node

# Export kubeconfig for later use
mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config
```

Install the NVIDIA GPU Operator via Helm — this makes GPUs schedulable resources in K8s:

```bash
# Helm
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
kubectl create namespace gpu-operator
helm install gpu-operator nvidia/gpu-operator \
    -n gpu-operator \
    --set driver.enabled=false \
    --set toolkit.enabled=true \
    --wait
```

Driver is disabled because Lambda's image already provides it — installing again causes conflicts.

**Pass gate:**

```bash
kubectl get pods -n gpu-operator      # all Running or Completed
kubectl describe node | grep -A2 "Capacity:" | grep "nvidia.com/gpu"
# Should show: nvidia.com/gpu: 1
```

---

## Phase 2 — Install HAMi for fractional GPU sharing

**Time:** 15 min. **Cost so far:** ~$2.

HAMi replaces the default NVIDIA device plugin with one that exposes fractional GPU memory and compute. This is what lets 5 pods share the H100.

```bash
# Get node name for scheduler label
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl label node $NODE gpu=on

# Install HAMi
helm repo add hami-charts https://project-hami.github.io/HAMi/
helm repo update
kubectl create namespace hami-system

helm install hami hami-charts/hami \
    -n hami-system \
    --set scheduler.kubeScheduler.imageTag=v1.29.0 \
    --wait
```

**Pass gate:**

```bash
kubectl get pods -n hami-system    # all Running
kubectl describe node $NODE | grep -A3 Capacity | grep -E "gpu|nvidia"
# Should now show BOTH:
#   nvidia.com/gpu: 1
#   nvidia.com/gpumem: 81920      (in MB)
#   nvidia.com/gpucores: 100      (percent)
```

Test with a canary pod:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: hami-canary
spec:
  restartPolicy: Never
  schedulerName: hami-scheduler
  containers:
  - name: cuda
    image: nvidia/cuda:12.4.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
        nvidia.com/gpumem: 8192
        nvidia.com/gpucores: 20
EOF

kubectl logs hami-canary   # should show a GPU with only 8GB visible
kubectl delete pod hami-canary
```

If the canary shows an 8 GB HBM limit even though the H100 has 80 GB, HAMi is working.

---

## Phase 3 — Observability stack (Prometheus + Grafana)

**Time:** 15 min. **Cost so far:** ~$3.

We install this early because we want metrics from every subsequent component.

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace observability

helm install monitor prometheus-community/kube-prometheus-stack \
    -n observability \
    --set grafana.adminPassword=admin \
    --set prometheus.prometheusSpec.scrapeInterval=15s \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --wait
```

Port-forward Grafana to your laptop (in a separate terminal on your laptop, not the node):

```bash
# On laptop
ssh -i $SSH_KEY -L 3000:localhost:3000 ubuntu@$LAMBDA_IP
# Then on the node:
kubectl port-forward -n observability svc/monitor-grafana 3000:80
```

Open `http://localhost:3000` (admin / admin) and confirm Prometheus data source is green.

Import the vLLM Grafana dashboard (dashboard ID 21833 from grafana.com, or paste the JSON from vLLM's repo).

**Pass gate:** Grafana shows GPU metrics from DCGM exporter (temperature, utilization) — proves the full metrics pipeline works before we add LLM workloads.

---

## Phase 4 — Deploy Mooncake KV storage tier

**Time:** 25 min. **Cost so far:** ~$4.

Mooncake stores KV cache blocks in CPU RAM + local NVMe, accessible from any pod. On a single node this is somewhat artificial (the "cluster" is one machine), but it exercises the plumbing and proves the integration works.

```bash
kubectl create namespace kv-tier

# Create a hostPath PVC on the node's NVMe for the persistent tier
# (Lambda instances come with a large ephemeral NVMe at /home)
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mooncake-nvme
spec:
  capacity:
    storage: 200Gi
  accessModes: ["ReadWriteOnce"]
  hostPath:
    path: /home/ubuntu/mooncake-storage
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-nvme
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mooncake-nvme
  namespace: kv-tier
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: local-nvme
  resources:
    requests:
      storage: 200Gi
EOF

mkdir -p /home/ubuntu/mooncake-storage

# Deploy Mooncake master + one storage node
# Using the reference container from Moonshot's repo
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mooncake-master
  namespace: kv-tier
spec:
  replicas: 1
  selector: { matchLabels: { app: mooncake, role: master } }
  template:
    metadata:
      labels: { app: mooncake, role: master }
    spec:
      containers:
      - name: master
        image: mooncakelabs/mooncake:v0.3.5
        args: ["--role=master", "--port=50051"]
        ports: [{ containerPort: 50051 }]
        resources:
          requests: { cpu: "500m", memory: "2Gi" }
          limits:   { cpu: "2",    memory: "4Gi" }
---
apiVersion: v1
kind: Service
metadata:
  name: mooncake-master
  namespace: kv-tier
spec:
  selector: { app: mooncake, role: master }
  ports: [{ port: 50051, targetPort: 50051 }]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mooncake-store
  namespace: kv-tier
spec:
  replicas: 1
  selector: { matchLabels: { app: mooncake, role: store } }
  template:
    metadata:
      labels: { app: mooncake, role: store }
    spec:
      containers:
      - name: store
        image: mooncakelabs/mooncake:v0.3.5
        args:
          - "--role=store"
          - "--master=mooncake-master.kv-tier.svc:50051"
          - "--cpu-mem-gb=32"
          - "--nvme-path=/data"
        volumeMounts:
        - { name: nvme, mountPath: /data }
        resources:
          requests: { cpu: "2", memory: "36Gi" }
          limits:   { cpu: "4", memory: "40Gi" }
      volumes:
      - name: nvme
        persistentVolumeClaim: { claimName: mooncake-nvme }
EOF
```

**Note on Mooncake versioning:** the tag `v0.3.5` is illustrative — check github.com/kvcache-ai/Mooncake for the current release tag before running. The image path may also be different from what I wrote — verify with `docker pull` first before applying the manifest, or the pods will `ImagePullBackOff`.

**Pass gate:**

```bash
kubectl get pods -n kv-tier   # both Running
kubectl logs -n kv-tier deploy/mooncake-master | grep -i ready
```

---

## Phase 5 — Deploy vLLM prefill + decode pods

**Time:** 30 min. **Cost so far:** ~$6.

Now the compute plane. We deploy:
- **2 prefill pods**, each with 15 GB HBM + 25% compute slice
- **3 decode pods**, each with 12 GB HBM + 15% compute slice
- Both pool types use vLLM 0.9+ with the Mooncake connector enabled

HBM math: 5 pods × (~13 GB weights + ~3 GB KV pool) ≈ 80 GB, right at the H100's ceiling. This is intentional — we want to see KV pressure.

```bash
kubectl create namespace inference

# Pull HuggingFace token as secret
kubectl create secret generic hf-token \
    -n inference \
    --from-literal=token=$HF_TOKEN

# Shared configmap for Mooncake connector
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: mooncake-config
  namespace: inference
data:
  mooncake.json: |
    {
      "master_address": "mooncake-master.kv-tier.svc.cluster.local:50051",
      "role": "client",
      "prefill_ports": ["mooncake-store.kv-tier.svc.cluster.local:50052"],
      "cpu_memory_gb": 32,
      "device_name": "cuda:0"
    }
EOF
```

Deploy the prefill pool:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prefill
  namespace: inference
spec:
  replicas: 2
  selector: { matchLabels: { app: vllm, role: prefill } }
  template:
    metadata:
      labels: { app: vllm, role: prefill }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      schedulerName: hami-scheduler
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.9.0
        args:
          - "--model=Qwen/Qwen2.5-7B-Instruct"
          - "--served-model-name=qwen7b"
          - "--host=0.0.0.0"
          - "--port=8000"
          - "--max-model-len=4096"
          - "--gpu-memory-utilization=0.75"
          - "--enable-prefix-caching"
          - "--kv-transfer-config"
          - '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
        env:
        - name: HF_TOKEN
          valueFrom: { secretKeyRef: { name: hf-token, key: token } }
        - name: MOONCAKE_CONFIG_PATH
          value: /config/mooncake.json
        ports: [{ containerPort: 8000, name: http }]
        volumeMounts:
        - { name: config, mountPath: /config }
        - { name: hf-cache, mountPath: /root/.cache/huggingface }
        resources:
          limits:
            nvidia.com/gpu: 1
            nvidia.com/gpumem: 15360      # 15 GB
            nvidia.com/gpucores: 25       # 25% SMs
      volumes:
      - name: config
        configMap: { name: mooncake-config }
      - name: hf-cache
        hostPath: { path: /home/ubuntu/hf-cache, type: DirectoryOrCreate }
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-prefill
  namespace: inference
  labels: { app: vllm, role: prefill }
spec:
  selector: { app: vllm, role: prefill }
  ports: [{ port: 8000, targetPort: 8000 }]
EOF
```

Deploy the decode pool (same manifest, different `kv_role`, smaller slices, 3 replicas):

```bash
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-decode
  namespace: inference
spec:
  replicas: 3
  selector: { matchLabels: { app: vllm, role: decode } }
  template:
    metadata:
      labels: { app: vllm, role: decode }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      schedulerName: hami-scheduler
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.9.0
        args:
          - "--model=Qwen/Qwen2.5-7B-Instruct"
          - "--served-model-name=qwen7b"
          - "--host=0.0.0.0"
          - "--port=8000"
          - "--max-model-len=4096"
          - "--gpu-memory-utilization=0.75"
          - "--enable-prefix-caching"
          - "--kv-transfer-config"
          - '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
        env:
        - name: HF_TOKEN
          valueFrom: { secretKeyRef: { name: hf-token, key: token } }
        - name: MOONCAKE_CONFIG_PATH
          value: /config/mooncake.json
        ports: [{ containerPort: 8000, name: http }]
        volumeMounts:
        - { name: config, mountPath: /config }
        - { name: hf-cache, mountPath: /root/.cache/huggingface }
        resources:
          limits:
            nvidia.com/gpu: 1
            nvidia.com/gpumem: 12288      # 12 GB
            nvidia.com/gpucores: 15       # 15% SMs
      volumes:
      - name: config
        configMap: { name: mooncake-config }
      - name: hf-cache
        hostPath: { path: /home/ubuntu/hf-cache, type: DirectoryOrCreate }
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-decode
  namespace: inference
  labels: { app: vllm, role: decode }
spec:
  selector: { app: vllm, role: decode }
  ports: [{ port: 8000, targetPort: 8000 }]
EOF
```

First pod boot will take 5-10 minutes (model download). Subsequent pods reuse the hostPath cache.

**Pass gate:**

```bash
kubectl -n inference get pods -w
# Wait until 2 prefill + 3 decode pods all Running with 1/1 Ready

# Test one prefill pod directly
kubectl -n inference port-forward svc/vllm-prefill 8001:8000 &
curl http://localhost:8001/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen7b","prompt":"The capital of France is","max_tokens":10}'
```

Expect a completion with "Paris" and 200 OK.

**If Mooncake connector fails to initialize:** vLLM will fall back to normal operation but you'll lose the KV transfer feature. In that case, drop the `--kv-transfer-config` flags and continue with plain prefix caching for the first pass of benchmarks — you can revisit Mooncake later. The exact connector arguments have moved between vLLM releases; check `vllm serve --help | grep -A5 kv-transfer` on your version.

---

## Phase 6 — Deploy SGLang router (cache-aware routing)

**Time:** 10 min. **Cost so far:** ~$8.

The SGLang router does prefix-hash routing — sends requests with shared prefixes to the same backend pod, maximizing prefix cache hits.

```bash
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sgl-router
  namespace: inference
spec:
  replicas: 1
  selector: { matchLabels: { app: router } }
  template:
    metadata:
      labels: { app: router }
    spec:
      containers:
      - name: router
        image: lmsysorg/sglang:v0.4.6-router
        args:
          - "python"
          - "-m"
          - "sglang_router.launch_router"
          - "--host=0.0.0.0"
          - "--port=8080"
          - "--worker-urls"
          - "http://vllm-decode.inference.svc.cluster.local:8000"
          - "--policy=cache_aware"
        ports: [{ containerPort: 8080 }]
        resources:
          requests: { cpu: "500m", memory: "512Mi" }
          limits:   { cpu: "1",    memory: "1Gi" }
---
apiVersion: v1
kind: Service
metadata:
  name: sgl-router
  namespace: inference
spec:
  selector: { app: router }
  ports: [{ port: 8080, targetPort: 8080 }]
EOF
```

Note: SGLang router doesn't natively understand "prefill vs decode pools" — for cleaner disaggregation, llm-d would be a better fit, but its Helm charts churn quickly. SGLang router is more stable and does prefix-aware routing which is 80% of the value. If you want to try llm-d instead, see the addendum at the end.

### Router alternatives and the admission queue

The runnable implementation (`blueprint1/`) keeps a single `router` Service on :8080 and puts **one of two** router implementations behind it:

- **`sglang_router`** (config B, default): the router above. It does cache-aware prefix routing to the decode pods only, and has **no queue**. Every request goes to a pod right away and waits in that vLLM pod's own scheduler queue.
- **`kv_router`** (config C; also config B with `ROUTER_IMPL_B=kv`): a small bundled router with the same cache-aware policy. It also runs the prefill → decode hop, which the SGLang setup above can't do, and adds an **admission queue**.

Why a queue: without one, a burst is split across pods the moment it arrives. A request stuck behind a busy pod can't move to a pod that frees up sooner, nothing pushes back on clients, and the backlog is spread across per-pod queues where the router can't see it. The admission queue fixes that:

- Each decode pod gets at most `ROUTER_MAX_INFLIGHT_PER_WORKER` requests at once (default `MAX_NUM_SEQS` = 64). The rest wait in **one bounded FIFO** in the router.
- The pod is picked **when a slot frees up**, not when the request arrives, still preferring the pod with the longest matching prefix.
- A full queue (`ROUTER_QUEUE_MAX`, 512) returns **429 + Retry-After**. A request waiting longer than `ROUTER_QUEUE_TIMEOUT` (120 s) gets a **503**.
- Metrics: `kv_router_queue_depth`, `kv_router_rejected_total{reason}`, `kv_router_queue_wait_seconds_*`.

This is an in-memory queue for interactive, streamed requests, not a message broker. A durable queue (Kafka/Redis) only makes sense for an offline batch endpoint.

**Pass gate:**

```bash
kubectl -n inference port-forward svc/sgl-router 8080:8080 &
curl http://localhost:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen7b","prompt":"Hello","max_tokens":10}'
```

---

## Phase 7 — Deploy LiteLLM gateway

**Time:** 10 min. **Cost so far:** ~$9.

LiteLLM is the outermost layer — auth, rate limiting, unified API. It fronts the router.

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-config
  namespace: inference
data:
  config.yaml: |
    model_list:
      - model_name: qwen7b
        litellm_params:
          model: openai/qwen7b
          api_base: http://sgl-router.inference.svc:8080/v1
          api_key: dummy
    litellm_settings:
      set_verbose: false
      cache: true
    general_settings:
      master_key: sk-lambda-demo
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm
  namespace: inference
spec:
  replicas: 1
  selector: { matchLabels: { app: litellm } }
  template:
    metadata: { labels: { app: litellm } }
    spec:
      containers:
      - name: litellm
        image: ghcr.io/berriai/litellm:main-stable
        args: ["--config=/config/config.yaml", "--port=4000"]
        ports: [{ containerPort: 4000 }]
        volumeMounts: [{ name: config, mountPath: /config }]
      volumes:
      - name: config
        configMap: { name: litellm-config }
---
apiVersion: v1
kind: Service
metadata:
  name: litellm
  namespace: inference
spec:
  type: NodePort
  selector: { app: litellm }
  ports:
  - port: 4000
    targetPort: 4000
    nodePort: 30400
EOF
```

**Pass gate:** end-to-end test from your laptop:

```bash
curl http://$LAMBDA_IP:30400/v1/completions \
  -H 'Authorization: Bearer sk-lambda-demo' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen7b","prompt":"1+1=","max_tokens":5}'
```

Every layer is now up: `laptop → LiteLLM (auth) → SGLang router (cache routing) → vLLM decode pool → (KV via Mooncake) ← vLLM prefill pool`. In the runnable implementation, config C uses `laptop → LiteLLM → kv_router (admission queue → cache routing → prefill hop) → vLLM decode pool` instead.

---

## Phase 8 — Benchmark grid

**Time:** ~4 hours. **Cost:** ~$10.

We run four arrival patterns × three configurations, capturing TTFT/ITL/throughput/KV metrics for each.

### Configurations

| Config | Description | How to enable |
|---|---|---|
| **A** — Baseline single pod | 1 vLLM pod, no prefix caching, no router | Direct-hit one pod; disable prefix caching |
| **B** — Prefix-cached, routed | Full 3-decode-pod pool with SGLang router, prefix caching on, no Mooncake | Current setup, minus Mooncake |
| **C** — Full disaggregation | Everything: prefill + decode + Mooncake + router | Current setup as-is |

In the runnable implementation, only config C's router (`kv_router`) has the admission queue, so B → C also includes the queue's effect, especially at `rate=inf`. Set `ROUTER_IMPL_B=kv` to give B the same router and queue, or set `ROUTER_MAX_INFLIGHT_PER_WORKER=0` to turn the queue off in C.

### Arrival patterns (per config)

For each config, run:

1. **rate=1, 200 prompts** — low utilization baseline
2. **rate=8, 200 prompts** — moderate load
3. **rate=inf, 200 prompts** — burst
4. **Multi-turn simulation** — 50 sessions × 5 turns each with growing context (this is where Mooncake should shine)

### Benchmark script

Create on the node:

```bash
mkdir -p ~/bench && cd ~/bench

# vllm bench comes with the vLLM image; install client-side too
pip install vllm==0.9.0 numpy pandas

# Config A: temporary single-pod deployment
# (Scale current deploys to 0, spin up one clean pod)
```

Standard benchmark invocations look like this — record output JSON for each:

```bash
# Set the endpoint (via NodePort to LiteLLM, or port-forward to router)
export ENDPOINT=http://localhost:8080

for RATE in 1 8 inf; do
  vllm bench serve \
    --backend openai \
    --base-url $ENDPOINT \
    --model qwen7b \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 200 \
    --request-rate $RATE \
    --save-result \
    --result-filename results_C_rate${RATE}.json
done
```

For the multi-turn simulation, use vLLM's ShareGPT dataset mode (approximates real conversations):

```bash
# Download ShareGPT-style dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --backend openai \
  --base-url $ENDPOINT \
  --model qwen7b \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 250 \
  --request-rate 4 \
  --save-result \
  --result-filename results_C_sharegpt.json
```

### Collect vLLM metrics alongside

While each run happens, snapshot `/metrics` from a decode pod to capture the server-side view:

```bash
# In a second terminal, during each run:
kubectl -n inference exec deploy/vllm-decode -- curl -s localhost:8000/metrics \
  > metrics_C_rate${RATE}.txt
```

### Ablation collection

After all runs, you should have 12 JSON files + 12 metrics dumps. What to compare across configs:

| Metric | Where to find it | What "good" looks like |
|---|---|---|
| TTFT p95 | Benchmark JSON | Lower is better; expect C ≈ B > A |
| ITL p95 | Benchmark JSON | Lower is better; A ≈ B ≈ C |
| Output tok/s | Benchmark JSON | Higher is better; C > B > A |
| Prefix cache hit % | Metrics dump (`prefix_cache_hits_total / prefix_cache_queries_total`) | Much higher for C on multi-turn |
| Preemptions | Metrics dump (`num_preemptions_total`) | 0 is ideal |
| Mooncake KV transfer count | Mooncake master logs | Non-zero for C only |

The interesting comparison you'll actually see:

- **A → B:** big jump in throughput on high concurrency runs (batching + prefix caching)
- **B → C:** modest gain on random-token runs, **large gain on multi-turn/ShareGPT run** (where Mooncake wins on prefix reuse across pods)

If you don't see B → C improvement on multi-turn, either Mooncake isn't connected (check master logs for actual transfers) or the router isn't sticking follow-up turns to the right pod. Both are fixable, but debugging burns time — timebox to 60 minutes before dropping to "config C = config B with a note that Mooncake integration is left as future work".

### Cost checkpoint

At end of Phase 8, spend should be ~$25 total ($2.50/hr × ~10 hours of active work). Well within budget.

---

## Phase 9 — Tear down or checkpoint

**Time:** 5 min.

When stopping for the day:

```bash
# On laptop
scp -i $SSH_KEY -r ubuntu@$LAMBDA_IP:~/bench ./results-single-node
# Then terminate the instance from Lambda console (not just stop — Lambda bills for stopped instances too)
```

If you plan to resume within a few hours, `sudo systemctl stop k3s` keeps everything on disk and you can restart later — but the instance still costs $2.50/hr while running.

---

## Deliverables from single-node phase

- 12 benchmark JSON files (3 configs × 4 arrival patterns)
- 12 metrics snapshots
- A short results table comparing configs on TTFT p95 / ITL p95 / tok/s / cache hit rate
- One clean shutdown

Once this works and you have baseline numbers, proceed to Blueprint 2 for the multi-GPU version.

---

## Addendum — swapping SGLang router for llm-d

If you want to try llm-d instead (better prefill/decode awareness), the sketch is:

```bash
helm repo add llm-d https://llm-d.github.io/llm-d
helm install llm-d llm-d/llm-d -n inference \
    --set inferenceService.model=qwen7b \
    --set prefill.deployment=vllm-prefill \
    --set decode.deployment=vllm-decode
```

But: llm-d's Helm chart schema has changed several times in 2025-2026, and it depends on the Gateway API CRDs being installed. Budget an extra 1-2 hours to make it work. If SGLang router with prefix caching is producing measurable improvements on the multi-turn benchmark, that's enough evidence of layer value — don't burn hours on llm-d unless you specifically want to reproduce that stack.

---

## Common failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| HAMi canary shows full 80 GB | Pod scheduled by default scheduler, not HAMi | Add `schedulerName: hami-scheduler` to spec |
| vLLM pod OOMs on startup | Slice too small for 7B weights | Raise `nvidia.com/gpumem` to 16384+ |
| Model download times out | HF token missing or rate-limited | Verify secret, or pre-download once via `huggingface-cli` |
| Mooncake pods `ImagePullBackOff` | Tag doesn't exist | Check github.com/kvcache-ai/Mooncake for current release |
| SGLang router returns 500 | Backend healthcheck failed | Ensure vLLM pods report `/health` as 200 |
| Grafana shows no vLLM metrics | ServiceMonitor missing | Create ServiceMonitor for the inference namespace |
| Router doesn't stick prefixes | Not using `--policy=cache_aware` | Restart router with the flag |
| `kv_router` returns 429 / 503 | Admission queue full / wait timed out | Raise `ROUTER_QUEUE_MAX` / `ROUTER_QUEUE_TIMEOUT`, or reduce offered load. Check `/workers` on the router for pod health and `queue_depth` |

---

## What this proves and what it doesn't

**Proves:**
- Every layer of the architecture diagram can coexist and talk to each other
- HAMi genuinely lets multiple LLM pods share one GPU
- Cache-aware routing measurably improves multi-turn workloads
- The full observability path works end-to-end

**Doesn't prove:**
- Real disaggregated latency wins (all pods share HBM bandwidth on one GPU)
- Mooncake's cross-node RDMA path (there's no second node)
- Scaling behavior beyond ~5 concurrent pods per GPU
- Actual production throughput on a 70B+ model

Those require the multi-GPU setup — see Blueprint 2.
