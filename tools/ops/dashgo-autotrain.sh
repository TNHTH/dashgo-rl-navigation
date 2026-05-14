#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/gwh/dashgo_rl_project"
AUTOPILOT_ROOT="$PROJECT_ROOT/.artifacts/autopilot"
METRICS_ROOT="$AUTOPILOT_ROOT/metrics"
STATE_PATH="$METRICS_ROOT/continuous_supervisor_state.json"
EVENT_LOG_PATH="$METRICS_ROOT/continuous_supervisor_events.jsonl"
NOHUP_LOG="$METRICS_ROOT/continuous_gen2_supervisor.nohup.log"
PID_FILE="$METRICS_ROOT/continuous_gen2_supervisor.pid"
RUN_SCRIPT="$AUTOPILOT_ROOT/run_continuous_supervisor.sh"

usage() {
  cat <<'EOF'
用法:
  dashgo-autotrain.sh status
  dashgo-autotrain.sh watch [秒]
  dashgo-autotrain.sh report
  dashgo-autotrain.sh logs [supervisor|current] [-f]
  dashgo-autotrain.sh events [-f]
  dashgo-autotrain.sh pause
  dashgo-autotrain.sh resume
  dashgo-autotrain.sh start
  dashgo-autotrain.sh restart

说明:
  status   查看当前训练状态、活跃进程、当前 run 和关键路径
  watch    每隔 N 秒刷新 status，默认 5 秒
  report   输出一份适合人工查看的训练报告摘要
  logs     看 supervisor 日志或当前 run 日志；加 -f 持续跟随
  events   看事件流；加 -f 持续跟随
  pause    安全暂停：当前波次自然结束后停机
  resume    解除 safe pause，并确保 supervisor 运行
  start    如果未运行则启动 supervisor
  restart  重启 supervisor；若训练进程仍在，重启后会尝试接管
EOF
}

require_file() {
  local target="$1"
  if [[ ! -f "$target" ]]; then
    echo "缺少文件: $target" >&2
    exit 1
  fi
}

python_state() {
python3 - "$STATE_PATH" "$@" <<'PY'
import json
import pathlib
import subprocess
import sys

state_path = pathlib.Path(sys.argv[1])
cmd = sys.argv[2] if len(sys.argv) > 2 else "status"
metrics_root = state_path.parent

def load_state():
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))

def supervisor_pid():
    pid_file = metrics_root / "continuous_gen2_supervisor.pid"
    if not pid_file.exists():
        return None, False
    value = pid_file.read_text(encoding="utf-8").strip()
    if not value:
        return None, False
    try:
        pid = int(value)
    except ValueError:
        return value, False
    running = subprocess.run(["bash", "-lc", f"ps -p {pid} >/dev/null 2>&1"], check=False).returncode == 0
    return pid, running

def train_cmds():
    result = subprocess.run(
        ["bash", "-lc", "ps -eo cmd | rg 'train_v2.py --headless --gen gen2' || true"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "ps -eo cmd" in line or line.startswith("rg "):
            continue
        lines.append(line)
    return lines

state = load_state()
pid, pid_running = supervisor_pid()
train_processes = train_cmds()

if cmd == "status":
    print(f"supervisor_status: {state.get('supervisor_status')}")
    print(f"message: {state.get('message')}")
    print(f"active_run_name: {state.get('active_run_name')}")
    print(f"active_train_process_count(state): {state.get('active_train_process_count')}")
    print(f"active_train_process_count(real): {len(train_processes)}")
    print(f"last_heartbeat_at: {state.get('last_heartbeat_at')}")
    print(f"updated_at: {state.get('updated_at')}")
    print(f"next_trial: {state.get('next_trial')}")
    print(f"desired_state: {state.get('desired_state')}")
    print(f"pause_scope: {state.get('pause_scope')}")
    print(f"supervisor_pid: {pid}")
    print(f"supervisor_pid_running: {pid_running}")
    if train_processes:
        print("train_processes:")
        for item in train_processes:
            print(f"  - {item}")
    else:
        print("train_processes: []")
    print("paths:")
    print(f"  state: {state_path}")
    print(f"  events: {metrics_root / 'continuous_supervisor_events.jsonl'}")
    print(f"  supervisor_log: {metrics_root / 'continuous_gen2_supervisor.nohup.log'}")
    current_run = state.get("active_run_name")
    if isinstance(current_run, str) and current_run:
        print(f"  current_run_log: {metrics_root / f'{current_run}.log'}")
elif cmd == "report":
    print("# DashGo 自动训练报告")
    print(f"- 状态: {state.get('supervisor_status')}")
    print(f"- 说明: {state.get('message')}")
    print(f"- 当前 run: {state.get('active_run_name')}")
    print(f"- 当前心跳: {state.get('last_heartbeat_at')}")
    print(f"- 活动训练进程(真实): {len(train_processes)}")
    print(f"- 自动 follow-up 轮数: {state.get('auto_generated_rounds')}")
    print(f"- 下一步: {state.get('next_trial')}")
    summary = state.get("summary") or {}
    latest_scalars = summary.get("latest_scalars") or {}
    if latest_scalars:
        print("- 关键指标:")
        for key in [
            "Curriculum/target_adaptive",
            "Episode_Termination/reach_goal",
            "Episode_Termination/object_collision",
            "Episode_Termination/time_out",
            "Metrics/target_pose/position_error",
            "Metrics/target_pose/orientation_error",
            "Train/mean_reward",
        ]:
            if key in latest_scalars:
                print(f"  - {key}: {latest_scalars[key]}")
    current_run = state.get("active_run_name")
    if isinstance(current_run, str) and current_run:
        print(f"- 当前 run 日志: {metrics_root / f'{current_run}.log'}")
    print(f"- Supervisor 日志: {metrics_root / 'continuous_gen2_supervisor.nohup.log'}")
    print(f"- 事件流: {metrics_root / 'continuous_supervisor_events.jsonl'}")
else:
    raise SystemExit(f"不支持的子命令: {cmd}")
PY
}

set_pause_state() {
  python3 - "$STATE_PATH" <<'PY'
import json
import pathlib
import time
import sys

path = pathlib.Path(sys.argv[1])
state = {}
if path.exists():
    state = json.loads(path.read_text(encoding="utf-8"))
state["desired_state"] = "pause_after_current_run"
state["pause_scope"] = "all"
state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
}

clear_pause_state() {
  python3 - "$STATE_PATH" <<'PY'
import json
import pathlib
import time
import sys

path = pathlib.Path(sys.argv[1])
state = {}
if path.exists():
    state = json.loads(path.read_text(encoding="utf-8"))
state["desired_state"] = None
state["pause_scope"] = None
state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
}

current_run_log() {
python3 - "$STATE_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
state = json.loads(path.read_text(encoding="utf-8"))
run_name = state.get("active_run_name")
if not isinstance(run_name, str) or not run_name:
    raise SystemExit(1)
print(path.parent / f"{run_name}.log")
PY
}

cmd="${1:-status}"
shift || true

case "$cmd" in
  status)
    require_file "$STATE_PATH"
    python_state status
    ;;
  watch)
    interval="${1:-5}"
    while true; do
      clear
      "$0" status
      echo
      echo "---- supervisor 日志尾部 ----"
      tail -n 12 "$NOHUP_LOG" 2>/dev/null || true
      sleep "$interval"
    done
    ;;
  report)
    require_file "$STATE_PATH"
    python_state report
    ;;
  logs)
    target="${1:-supervisor}"
    follow="${2:-}"
    if [[ "$target" == "-f" ]]; then
      target="supervisor"
      follow="-f"
    fi
    if [[ "$target" == "supervisor" ]]; then
      if [[ "$follow" == "-f" ]]; then
        tail -f "$NOHUP_LOG"
      else
        tail -n 80 "$NOHUP_LOG"
      fi
    elif [[ "$target" == "current" ]]; then
      log_path="$(current_run_log || true)"
      if [[ -z "${log_path:-}" || ! -f "$log_path" ]]; then
        echo "当前没有可用的 run 日志。" >&2
        exit 1
      fi
      if [[ "$follow" == "-f" ]]; then
        tail -f "$log_path"
      else
        tail -n 80 "$log_path"
      fi
    else
      echo "未知日志目标: $target" >&2
      exit 1
    fi
    ;;
  events)
    follow="${1:-}"
    if [[ "$follow" == "-f" ]]; then
      tail -f "$EVENT_LOG_PATH"
    else
      tail -n 80 "$EVENT_LOG_PATH"
    fi
    ;;
  pause)
    set_pause_state >/dev/null
    echo "已写入 safe pause 请求。当前波次结束后会停机。"
    ;;
  resume|start)
    clear_pause_state >/dev/null
    "$RUN_SCRIPT"
    echo "已解除暂停并确保 supervisor 运行。"
    ;;
  restart)
    clear_pause_state >/dev/null
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
        kill "$pid"
        sleep 1
      fi
    fi
    "$RUN_SCRIPT"
    echo "已重启 supervisor。若训练进程仍在，supervisor 会尝试接管。"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
