#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${BUNDLE_ROOT}/eval_outputs}"
LOG_DIR="${EVAL_OUTPUT_ROOT}/logs"
PID_FILE="${EVAL_OUTPUT_ROOT}/launcher.pid"
mkdir -p "${LOG_DIR}"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Evaluation suite appears to be running: pid=${old_pid}"
    echo "Check ${LOG_DIR}/master.log"
    exit 1
  fi
fi

nohup bash "${SCRIPT_DIR}/run_all.sh" >> "${LOG_DIR}/master.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"
echo "Evaluation suite launched: pid=${pid}"
echo "Log: ${LOG_DIR}/master.log"
echo "Status: ${EVAL_OUTPUT_ROOT}/STATUS.tsv"
