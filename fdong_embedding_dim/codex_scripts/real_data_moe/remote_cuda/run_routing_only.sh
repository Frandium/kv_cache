#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_distributed proposed 2048 2048 "${OUTPUT_ROOT}/proposed_no_lb" 0.0
