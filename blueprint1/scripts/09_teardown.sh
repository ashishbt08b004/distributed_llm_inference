#!/usr/bin/env bash
# Phase 9 — checkpoint or tear down (run ON the node).
#
#   ./09_teardown.sh pause     stop k3s (everything stays on disk; instance still bills!)
#   ./09_teardown.sh resume    start k3s again
#   ./09_teardown.sh workloads delete the inference + kv-tier namespaces, keep the cluster
#   ./09_teardown.sh nuke      uninstall k3s entirely (results in ~/bench are kept)
#
# The only way to stop paying is to TERMINATE the instance in the Lambda console
# after fetching results with `./scripts/laptop.sh fetch` from your laptop.
source "$(dirname "$0")/lib.sh"

case "${1:-}" in
  pause)     sudo systemctl stop k3s; log "k3s stopped. Instance is still running and billing." ;;
  resume)    sudo systemctl start k3s; wait_until "node Ready" 180 bash -c "kubectl get nodes | grep -q ' Ready'"; kubectl get pods -A | grep -v Running || true ;;
  workloads) kubectl delete ns inference kv-tier --ignore-not-found; kubectl delete pv mooncake-nvme --ignore-not-found; rm -f "$BP1_ROOT/.active_config" ;;
  nuke)      /usr/local/bin/k3s-uninstall.sh; log "k3s removed. Results remain in $BENCH_DIR" ;;
  *)         sed -n '2,10p' "$0"; exit 1 ;;
esac
