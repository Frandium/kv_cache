#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_distributed proposed 1536 1536 "${OUTPUT_ROOT}/proposed" "${M_PROPOSED_LB_WEIGHT:-0.01}"
