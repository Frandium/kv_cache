#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_distributed baseline 3072 3072 "${OUTPUT_ROOT}/baseline"
