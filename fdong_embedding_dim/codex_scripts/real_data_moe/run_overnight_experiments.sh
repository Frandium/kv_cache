#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

bash fdong_embedding_dim/codex_scripts/real_data_moe/run_proposed_routing_only_matched.sh
bash fdong_embedding_dim/codex_scripts/real_data_moe/run_large_batch_baseline_then_proposed.sh

echo "[done] all overnight experiments completed"
