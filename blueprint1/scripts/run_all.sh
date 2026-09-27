#!/usr/bin/env bash
# Run phases 0-7 in order (run ON the node). Each phase stops at its pass gate
# on failure, so you can fix things and re-run from that phase:
#   ./scripts/run_all.sh          # everything
#   ./scripts/run_all.sh 4        # start at phase 4
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
FROM="${1:-0}"
PHASES=(00_preflight 01_install_k3s_gpu 02_install_hami 03_install_observability
        04_deploy_mooncake 05_deploy_vllm 06_deploy_router 07_deploy_gateway)
for i in "${!PHASES[@]}"; do
  (( i < FROM )) && continue
  echo; echo "################ Phase $i: ${PHASES[$i]} ################"
  "$DIR/${PHASES[$i]}.sh"
done
echo; echo "Stack is up. Next: ./scripts/08_run_benchmarks.sh"
