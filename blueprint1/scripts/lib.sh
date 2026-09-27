# shellcheck shell=bash
# Common helpers, sourced by every phase script.
set -euo pipefail

BP1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BP1_ROOT

set -a
# shellcheck source=../config.env
source "$BP1_ROOT/config.env"
# Generated secrets (gateway key) persist here across runs.
[ -f "$BP1_ROOT/.secrets.env" ] && source "$BP1_ROOT/.secrets.env"
set +a

# Derived values shared by manifests.
export MOONCAKE_MASTER_HOST="mooncake-master.kv-tier.svc.cluster.local"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

c_green=$'\033[32m'; c_red=$'\033[31m'; c_yel=$'\033[33m'; c_off=$'\033[0m'
log()  { echo "${c_green}==>${c_off} $*"; }
warn() { echo "${c_yel}WARN:${c_off} $*" >&2; }
die()  { echo "${c_red}ERROR:${c_off} $*" >&2; exit 1; }
pass() { echo "${c_green}PASS GATE:${c_off} $*"; }

need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found — run the earlier phases first"; }

# Render a manifest template: substitutes only ${VARS} that are set in the
# environment, so anything else containing '$' is left untouched.
render() {
  local vars
  vars="$(env | cut -d= -f1 | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' | sed 's/^/$/' | tr '\n' ' ')"
  envsubst "$vars" < "$1"
}
apply_tpl() { render "$1" | kubectl apply -f -; }

# wait_until "<description>" <timeout-seconds> <command...>
wait_until() {
  local desc="$1" timeout="$2"; shift 2
  local start=$SECONDS
  log "Waiting for: $desc (timeout ${timeout}s)"
  until "$@" >/dev/null 2>&1; do
    (( SECONDS - start > timeout )) && die "Timed out waiting for: $desc"
    sleep 5
  done
}

# Wait for a StatefulSet/Deployment to have all replicas ready.
wait_rollout() {
  local ns="$1" kind_name="$2" timeout="${3:-900}"
  log "Waiting for $kind_name in $ns to be ready (timeout ${timeout}s)"
  kubectl -n "$ns" rollout status "$kind_name" --timeout="${timeout}s"
}

ensure_ns() { kubectl get ns "$1" >/dev/null 2>&1 || kubectl create namespace "$1"; }

ensure_gateway_key() {
  if [ -z "${LITELLM_MASTER_KEY:-}" ]; then
    LITELLM_MASTER_KEY="sk-bp1-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    echo "LITELLM_MASTER_KEY=\"$LITELLM_MASTER_KEY\"" >> "$BP1_ROOT/.secrets.env"
    chmod 600 "$BP1_ROOT/.secrets.env"
    export LITELLM_MASTER_KEY
    log "Generated gateway key into .secrets.env"
  fi
}

# Per-pod URLs of a vLLM StatefulSet (stable DNS through its headless service).
pod_urls() {
  local name="$1" replicas="$2" port="${3:-8000}" out=() i
  for ((i = 0; i < replicas; i++)); do
    out+=("http://${name}-${i}.${name}.inference.svc.cluster.local:${port}")
  done
  echo "${out[*]}"
}
