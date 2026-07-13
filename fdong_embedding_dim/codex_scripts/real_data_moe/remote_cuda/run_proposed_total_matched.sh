#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_distributed proposed 1536 3456 "${OUTPUT_ROOT}/proposed_total_matched"
