#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
RUN_DIR="${OUTPUT_ROOT}/baseline"
mkdir -p "${RUN_DIR}"
PID_FILE="${RUN_DIR}/launcher.pid"
if [[ -s "${PID_FILE}" ]]; then
  OLD_PID="$(<"${PID_FILE}")"
  if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "M baseline appears to be running already: pid=${OLD_PID}"
    echo "Refusing to launch a duplicate. Check ${RUN_DIR}/launcher.log"
    exit 1
  fi
fi
nohup bash "${SCRIPT_DIR}/run_baseline.sh" \
  > "${RUN_DIR}/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" > "${PID_FILE}"
echo "M baseline launched: pid=${PID} log=${RUN_DIR}/launcher.log"
