#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_distributed baseline 2048 2048 "${OUTPUT_ROOT}/baseline_lb" "${BASELINE_LB_WEIGHT:-0.01}"
