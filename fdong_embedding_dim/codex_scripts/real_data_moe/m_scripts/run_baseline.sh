#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_distributed baseline 1536 1536 "${OUTPUT_ROOT}/baseline" "${M_BASELINE_LB_WEIGHT:-0.01}"
