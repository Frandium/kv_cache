#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_distributed proposed 2048 2048 "${OUTPUT_ROOT}/proposed_lb" "${PROPOSED_LB_WEIGHT:-0.01}"
