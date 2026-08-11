#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# =========================
# Config Area
# =========================
# 直接在这里改模型规模与训练模式，然后执行：
#   ./run_train.sh
MODEL_SIZE="${MODEL_SIZE:-80B}"          # 30B | 80B
TRAIN_MODE="${TRAIN_MODE:-cross-token}"   # same-token | cross-token
# 统一数据源目录（四个 train/eval 脚本都会读取这个环境变量）
DATA_DOMAIN_DIR="${DATA_DOMAIN_DIR:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10}"
RUN_STAGE="${RUN_STAGE:-eval_only}" # train_only | eval_only | train_then_eval
EVAL_CKPT="${EVAL_CKPT:-/mnt/workspace/let-moe_predictor/checkpoints/DeepPredictor_Qwen3Next80B_deep2L_ft/0.last.pth}"               # 可选：指定 eval ckpt 绝对/相对路径
EVAL_TOPK="${EVAL_TOPK:-20}"               # 可选：覆盖 eval topk
EVAL_OUT_DIR="${EVAL_OUT_DIR:-../eval_outputs}"         # 可选：统一指定 eval 结果(npy)输出目录
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y%m%d%H%M)}"  # 评测文件时间戳，可手动覆盖

usage() {
  cat <<'EOF'
Usage:
  ./run_train.sh [target] [extra args passed to python script]

Targets:
  Qwen3_30B_same-token
  Qwen3_30B_cross-token
  Qwen3_80B_same-token
  Qwen3_80B_cross-token

Examples:
  ./run_train.sh
  DATA_DOMAIN_DIR=/mnt/workspace/your_data_dir ./run_train.sh
  EVAL_OUT_DIR=../eval_outputs ./run_train.sh Qwen3_80B_same-token
  RUN_STAGE=train_only ./run_train.sh Qwen3_30B_same-token
  RUN_STAGE=eval_only EVAL_CKPT=../checkpoints/xxx/0.last.pth ./run_train.sh Qwen3_80B_cross-token
  ./run_train.sh Qwen3_30B_same-token
  PYTHON_BIN=python3.10 ./run_train.sh Qwen3_80B_cross-token

When target is omitted:
  Use Config Area values to build target:
  Qwen3_${MODEL_SIZE}_${TRAIN_MODE}
EOF
}

export DATA_DOMAIN_DIR
if [[ -n "${EVAL_CKPT}" ]]; then
  export EVAL_CKPT
fi
if [[ -n "${EVAL_TOPK}" ]]; then
  export EVAL_TOPK
fi
if [[ -n "${EVAL_OUT_DIR}" ]]; then
  export EVAL_OUT_DIR
fi

if [[ "${1:-}" == "" ]]; then
  target="Qwen3_${MODEL_SIZE}_${TRAIN_MODE}"
else
  target="$1"
  shift || true
fi

train_script=""
eval_script=""
eval_tag=""

case "${target}" in
  Qwen3_30B_same-token)
    train_script="train_scripts/train_sametoken_Qwen30B.py"
    eval_script="eval_scripts/eval_sametoken_Qwen30B.py"
    eval_tag="sametoken30B"
    ;;
  Qwen3_30B_cross-token)
    train_script="train_scripts/train_crosstoken_Qwen30B.py"
    eval_script="eval_scripts/eval_crosstoken_Qwen30B.py"
    eval_tag="crosstoken30B"
    ;;
  Qwen3_80B_same-token)
    train_script="train_scripts/train_sametoken_Qwen80B.py"
    eval_script="eval_scripts/eval_sametoken_Qwen80B.py"
    eval_tag="sametoken80B"
    ;;
  Qwen3_80B_cross-token)
    train_script="train_scripts/train_crosstoken_Qwen80B.py"
    eval_script="eval_scripts/eval_crosstoken_Qwen80B.py"
    eval_tag="crosstoken80B"
    ;;
  *)
    echo "Unknown target: ${target}" >&2
    usage
    exit 2
    ;;
esac

cd "${PROJECT_DIR}"
export EVAL_TAG="${eval_tag}"
export EVAL_TIMESTAMP
echo "[run_train] target=${target} stage=${RUN_STAGE}"
echo "[run_train] train=${train_script}"
echo "[run_train] eval=${eval_script}"
echo "[run_train] eval_tag=${EVAL_TAG} eval_timestamp=${EVAL_TIMESTAMP}"

case "${RUN_STAGE}" in
  train_only)
    "${PYTHON_BIN}" "${train_script}" "$@"
    ;;
  eval_only)
    "${PYTHON_BIN}" "${eval_script}"
    ;;
  train_then_eval)
    "${PYTHON_BIN}" "${train_script}" "$@"
    "${PYTHON_BIN}" "${eval_script}"
    ;;
  *)
    echo "Unknown RUN_STAGE: ${RUN_STAGE}" >&2
    echo "Valid RUN_STAGE: train_only | eval_only | train_then_eval" >&2
    exit 3
    ;;
esac

