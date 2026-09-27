#!/usr/bin/env bash
# Switch the inference plane between ablation configs: ./set_config.sh A|B|C
source "$(dirname "$0")/stack.sh"
[ $# -eq 1 ] || die "usage: $0 A|B|C"
set_config "$1"
