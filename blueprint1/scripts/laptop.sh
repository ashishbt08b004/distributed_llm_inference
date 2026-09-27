#!/usr/bin/env bash
# Laptop-side helper. Needs LAMBDA_IP (and optionally SSH_KEY) exported.
#
#   ./scripts/laptop.sh sync       copy this repo to the node
#   ./scripts/laptop.sh ssh        open a shell on the node
#   ./scripts/laptop.sh tunnel     forward Grafana :3000, Prometheus :9090, gateway :4000 to localhost
#   ./scripts/laptop.sh smoke      end-to-end request through the gateway (via the tunnel)
#   ./scripts/laptop.sh fetch      copy ~/bench results to ../bench-results/{full,quick} (BENCH_RESULTS_DIR overrides)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../config.env
set -a; source "$ROOT/config.env"; [ -f "$ROOT/.secrets.env" ] && source "$ROOT/.secrets.env"; set +a
[ -n "${LAMBDA_IP:-}" ] || { echo "export LAMBDA_IP=<node public ip> first" >&2; exit 1; }
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$LAMBDA_IP")

case "${1:-}" in
  sync)
    rsync -az --delete -e "ssh -i $SSH_KEY" --exclude results-single-node --exclude .active_config --exclude .secrets.env \
      "$ROOT/" "ubuntu@$LAMBDA_IP:$REMOTE_REPO_DIR/"
    echo "synced to $LAMBDA_IP:$REMOTE_REPO_DIR" ;;
  ssh)
    exec "${SSH[@]}" ;;
  tunnel)
    echo "Grafana    http://localhost:3000  (admin/admin)"
    echo "Prometheus http://localhost:9090"
    echo "Gateway    http://localhost:4000  (Authorization: Bearer <LITELLM_MASTER_KEY>)"
    echo "Ctrl-C to close."
    exec "${SSH[@]}" -L 3000:localhost:13000 -L 9090:localhost:19090 -L 4000:localhost:30400 \
      'kubectl -n observability port-forward svc/monitor-grafana 13000:80 >/dev/null &
       kubectl -n observability port-forward svc/prometheus-operated 19090:9090 >/dev/null &
       trap "kill 0" EXIT; wait' ;;
  smoke)
    KEY="${LITELLM_MASTER_KEY:-$("${SSH[@]}" "grep LITELLM_MASTER_KEY $REMOTE_REPO_DIR/.secrets.env | cut -d'\"' -f2")}"
    curl -s http://localhost:4000/v1/chat/completions \
      -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=\"}],\"max_tokens\":5}"
    echo ;;
  fetch)
    # One results tree for every kind of benchmark, next to blueprint1/ and
    # analyst_crew/ (whose `run.py eval` writes to .../agent). Outside blueprint1/,
    # so `sync` never pushes results back to the node.
    RESULTS_ROOT="${BENCH_RESULTS_DIR:-$(cd "$ROOT/.." && pwd)/bench-results}"
    for d in results results-quick; do    # full grid -> full/, quick profile -> quick/
      dest="$RESULTS_ROOT/full"; [ "$d" = results-quick ] && dest="$RESULTS_ROOT/quick"
      if "${SSH[@]}" "test -d $BENCH_DIR/$d"; then
        mkdir -p "$dest"
        rsync -az -e "ssh -i $SSH_KEY" "ubuntu@$LAMBDA_IP:$BENCH_DIR/$d/" "$dest/"
        echo "$d -> $dest (see summary.md)"
      fi
    done ;;
  *)
    sed -n '2,9p' "$0"; exit 1 ;;
esac
