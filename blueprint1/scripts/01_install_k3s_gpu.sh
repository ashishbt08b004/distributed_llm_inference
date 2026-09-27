#!/usr/bin/env bash
# Phase 1 — k3s + NVIDIA container runtime + GPU Operator (run ON the node).
#
# Deviations from the blueprint text (and why):
#  * k3s is started with --default-runtime=nvidia. k3s auto-detects the host's
#    nvidia-container-runtime; making it the default is what HAMi requires.
#  * GPU Operator runs with driver/toolkit/devicePlugin DISABLED: the driver and
#    toolkit come from the Lambda image, and the device plugin comes from HAMi
#    (phase 2). Two device plugins advertising nvidia.com/gpu conflict.
#    The operator still gives us GPU feature discovery + the DCGM exporter.
source "$(dirname "$0")/lib.sh"

# --- NVIDIA container toolkit on the host (Lambda Stack usually has it) -------
if ! command -v nvidia-container-runtime >/dev/null; then
  log "Installing nvidia-container-toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit >/dev/null
fi
log "nvidia-container-runtime: $(command -v nvidia-container-runtime)"

# --- k3s -----------------------------------------------------------------------
if ! systemctl is-active --quiet k3s; then
  log "Installing k3s (traefik disabled, nvidia default runtime)"
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --disable=traefik --write-kubeconfig-mode=644 --default-runtime=nvidia" sh -
fi
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$USER" ~/.kube/config
chmod 600 ~/.kube/config
wait_until "k3s node Ready" 180 bash -c "kubectl get nodes | grep -q ' Ready'"
kubectl get nodes -o wide

# k3s creates RuntimeClasses for detected runtimes; make sure 'nvidia' exists.
kubectl get runtimeclass nvidia >/dev/null 2>&1 || kubectl apply -f "$BP1_ROOT/manifests/cluster/runtimeclass-nvidia.yaml"

# --- Helm ----------------------------------------------------------------------
if ! command -v helm >/dev/null; then
  log "Installing Helm"
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

# --- GPU Operator ----------------------------------------------------------------
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo update >/dev/null
ensure_ns gpu-operator
log "Installing NVIDIA GPU Operator"
helm upgrade --install gpu-operator nvidia/gpu-operator -n gpu-operator \
  ${GPU_OPERATOR_CHART_VERSION:+--version "$GPU_OPERATOR_CHART_VERSION"} \
  --set driver.enabled=false \
  --set toolkit.enabled=false \
  --set devicePlugin.enabled=false \
  --set migManager.enabled=false \
  --set operator.defaultRuntime=containerd \
  --wait --timeout 15m

# --- Pass gate ---------------------------------------------------------------------
kubectl get pods -n gpu-operator
log "Running a CUDA pod through the nvidia runtime"
kubectl delete pod gpu-smoke --ignore-not-found >/dev/null
kubectl run gpu-smoke --restart=Never --image="$CUDA_TEST_IMAGE" \
  --overrides='{"spec":{"runtimeClassName":"nvidia"}}' \
  --env=NVIDIA_VISIBLE_DEVICES=all -- nvidia-smi -L
wait_until "gpu-smoke pod to finish" 300 bash -c \
  "kubectl get pod gpu-smoke -o jsonpath='{.status.phase}' | grep -qE 'Succeeded|Failed'"
OUT=$(kubectl logs gpu-smoke)
kubectl delete pod gpu-smoke >/dev/null
echo "$OUT"
echo "$OUT" | grep -q "H100\|GPU 0" || die "Container could not see the GPU"
pass "k3s Ready, GPU Operator pods up, containers can see the GPU (nvidia.com/gpu appears after HAMi in phase 2)"
