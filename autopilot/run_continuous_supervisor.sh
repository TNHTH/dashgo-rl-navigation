#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/gwh/dashgo_rl_project"
AUTOPILOT_ROOT="$PROJECT_ROOT/.artifacts/autopilot"
LOG_FILE="$AUTOPILOT_ROOT/metrics/continuous_gen2_supervisor.nohup.log"
PID_FILE="$AUTOPILOT_ROOT/metrics/continuous_gen2_supervisor.pid"

mkdir -p "$AUTOPILOT_ROOT/metrics"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "already_running:$old_pid"
    exit 0
  fi
fi

cd "$PROJECT_ROOT"
nohup setsid stdbuf -oL -eL python3 autopilot/continuous_gen2_supervisor.py >>"$LOG_FILE" 2>&1 < /dev/null &
new_pid=$!
echo "$new_pid" > "$PID_FILE"
echo "$new_pid"
