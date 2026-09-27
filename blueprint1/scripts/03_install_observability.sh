#!/usr/bin/env bash
# Phase 3 — Prometheus + Grafana (kube-prometheus-stack), DCGM GPU metrics,
# and the monitors/dashboard for every component we deploy later.
source "$(dirname "$0")/lib.sh"
need helm

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null
ensure_ns observability

log "Installing kube-prometheus-stack"
# k3s runs etcd/scheduler/controller-manager/proxy inside the k3s binary, so their
# default scrape targets are always "down" — disable them to keep alerts quiet.
helm upgrade --install monitor prometheus-community/kube-prometheus-stack -n observability \
  ${KPS_CHART_VERSION:+--version "$KPS_CHART_VERSION"} \
  --set grafana.adminPassword=admin \
  --set prometheus.prometheusSpec.scrapeInterval=15s \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.retention=7d \
  --set kubeEtcd.enabled=false \
  --set kubeControllerManager.enabled=false \
  --set kubeScheduler.enabled=false \
  --set kubeProxy.enabled=false \
  --wait --timeout 15m

log "Enabling the DCGM exporter ServiceMonitor on the GPU Operator (CRD now exists)"
helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator --reuse-values \
  --set dcgmExporter.serviceMonitor.enabled=true --wait --timeout 10m

log "PodMonitors for vLLM / routers / Mooncake + the Grafana dashboard"
ensure_ns inference
ensure_ns kv-tier
kubectl apply -f "$BP1_ROOT/manifests/observability/podmonitors.yaml"
kubectl -n observability create configmap bp1-dashboard \
  --from-file=bp1-inference.json="$BP1_ROOT/manifests/observability/grafana-dashboard.json" \
  --dry-run=client -o yaml | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -

kubectl get pods -n observability

# --- Pass gate: DCGM GPU metrics are in Prometheus -----------------------------------
kubectl -n observability port-forward svc/prometheus-operated 19090:9090 >/dev/null 2>&1 &
PF=$!; trap 'kill $PF 2>/dev/null || true' EXIT
wait_until "DCGM_FI_DEV_GPU_UTIL in Prometheus" 300 bash -c \
  "curl -s 'http://localhost:19090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL' | jq -e '.data.result | length > 0'"
curl -s 'http://localhost:19090/api/v1/query?query=DCGM_FI_DEV_GPU_TEMP' | jq -r '.data.result[] | "GPU temp: \(.value[1]) C"'
pass "GPU metrics flow DCGM -> Prometheus. Open Grafana via the laptop tunnel (see README) — dashboard 'Blueprint 1 — Inference stack'."
