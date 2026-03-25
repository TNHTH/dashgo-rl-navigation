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
REGRESSION_STATE_PATH="$METRICS_ROOT/regression_state.json"
REGRESSION_EVENTS_PATH="$METRICS_ROOT/regression_events.jsonl"
REGRESSION_NOHUP_LOG="$METRICS_ROOT/training_regression.nohup.log"
REGRESSION_PID_FILE="$METRICS_ROOT/training_regression.pid"
REGRESSION_SUMMARY_JSON="$METRICS_ROOT/training_regression_gen2_formal.json"
AUTORESEARCH_ROOT="$AUTOPILOT_ROOT/autoresearch"
AUTORESEARCH_STATE_PATH="$AUTORESEARCH_ROOT/state.json"
AUTORESEARCH_EVENTS_PATH="$AUTORESEARCH_ROOT/events.jsonl"
AUTORESEARCH_NOHUP_LOG="$AUTORESEARCH_ROOT/autoresearch_supervisor.nohup.log"
AUTORESEARCH_PID_FILE="$AUTORESEARCH_ROOT/autoresearch_supervisor.pid"
AUTORESEARCH_KEEPALIVE_STATE_PATH="$AUTORESEARCH_ROOT/keepalive_state.json"
AUTORESEARCH_KEEPALIVE_EVENTS_PATH="$AUTORESEARCH_ROOT/keepalive_events.jsonl"
AUTORESEARCH_KEEPALIVE_NOHUP_LOG="$AUTORESEARCH_ROOT/autoresearch_keepalive.nohup.log"
AUTORESEARCH_KEEPALIVE_PID_FILE="$AUTORESEARCH_ROOT/autoresearch_keepalive.pid"
AUTORESEARCH_CMD=(
  python3.10
  "$PROJECT_ROOT/autopilot/autoresearch_supervisor.py"
  --project-root "$PROJECT_ROOT"
)
AUTORESEARCH_KEEPALIVE_CMD=(
  python3.10
  "$PROJECT_ROOT/tools/ops/autoresearch_keepalive.py"
  --project-root "$PROJECT_ROOT"
)
REGRESSION_CMD=(
  python3.10
  "$PROJECT_ROOT/tools/diagnostics/run_training_regression.py"
  --generation gen2
  --run-name-prefix bounded_tanh_regression
  --seeds 41,42,43
  --num-envs 8
  --max-iterations 9000
  --save-interval 100
  --suite main
  --requested-episodes 500
  --env-backoff 8,6,4
  --max-retries-per-seed 3
  --staging-export
  --summary-json "$REGRESSION_SUMMARY_JSON"
)

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
  dashgo-autotrain.sh regression-start
  dashgo-autotrain.sh regression-status
  dashgo-autotrain.sh regression-watch [秒]
  dashgo-autotrain.sh regression-logs [-f]
  dashgo-autotrain.sh regression-stop
  dashgo-autotrain.sh regression-resume
  dashgo-autotrain.sh autoresearch-start
  dashgo-autotrain.sh autoresearch-status
  dashgo-autotrain.sh autoresearch-watch [秒]
  dashgo-autotrain.sh autoresearch-logs [-f]
  dashgo-autotrain.sh autoresearch-pause
  dashgo-autotrain.sh autoresearch-resume
  dashgo-autotrain.sh autoresearch-stop
  dashgo-autotrain.sh autoresearch-report
  dashgo-autotrain.sh autoresearch-ensure-start [小时]
  dashgo-autotrain.sh autoresearch-ensure-status
  dashgo-autotrain.sh autoresearch-ensure-logs [-f]
  dashgo-autotrain.sh autoresearch-ensure-stop

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
  regression-start   后台启动正式三种子回归
                     可追加自定义参数，例如:
                     regression-start --run-name-prefix myrun --seeds 201 --max-iterations 19000
  regression-status  查看正式回归状态
  regression-watch   每隔 N 秒刷新正式回归状态，默认 5 秒
  regression-logs    查看正式回归 nohup 日志；加 -f 持续跟随
  regression-stop    停止正式回归后台进程
  regression-resume  从 regression_state.json 继续后台回归
  autoresearch-start   后台启动 autoresearch supervisor
                       可追加自定义参数，例如:
                       autoresearch-start --iteration-limit 3
  autoresearch-status  查看 autoresearch 状态
  autoresearch-watch   每隔 N 秒刷新 autoresearch 状态，默认 5 秒
  autoresearch-logs    查看 autoresearch nohup 日志；加 -f 持续跟随
  autoresearch-pause   请求 safe pause，当前研究轮结束后暂停
  autoresearch-resume  解除 safe pause，并确保 autoresearch supervisor 运行
  autoresearch-stop    停止 autoresearch supervisor
  autoresearch-report  输出一份适合人工查看的 autoresearch 摘要
  autoresearch-ensure-start  启动 keepalive，确保 autoresearch 至少持续指定小时数，默认 4 小时
  autoresearch-ensure-status 查看 keepalive 状态
  autoresearch-ensure-logs   查看 keepalive nohup 日志；加 -f 持续跟随
  autoresearch-ensure-stop   停止 keepalive 进程
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

python_regression_state() {
python3 - "$REGRESSION_STATE_PATH" "$REGRESSION_PID_FILE" "$@" <<'PY'
import json
import pathlib
import subprocess
import sys

state_path = pathlib.Path(sys.argv[1])
pid_path = pathlib.Path(sys.argv[2])
cmd = sys.argv[3] if len(sys.argv) > 3 else "status"

def load_state():
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))

def runner_pid():
    if not pid_path.exists():
        return None, False
    value = pid_path.read_text(encoding="utf-8").strip()
    if not value:
        return None, False
    try:
        pid = int(value)
    except ValueError:
        return value, False
    running = subprocess.run(["bash", "-lc", f"ps -p {pid} >/dev/null 2>&1"], check=False).returncode == 0
    return pid, running

state = load_state()
pid, running = runner_pid()

if cmd == "status":
    print(f"status: {state.get('status')}")
    print(f"message: {state.get('message')}")
    print(f"current_seed: {state.get('current_seed')}")
    print(f"current_run_name: {state.get('current_run_name')}")
    print(f"current_phase: {state.get('current_phase')}")
    print(f"current_attempt: {state.get('current_attempt')}")
    print(f"current_num_envs: {state.get('current_num_envs')}")
    print(f"current_log_path: {state.get('current_log_path')}")
    print(f"summary_path: {state.get('summary_path')}")
    print(f"updated_at: {state.get('updated_at')}")
    print(f"runner_pid: {pid}")
    print(f"runner_pid_running: {running}")
elif cmd == "report":
    print("# DashGo 正式回归状态")
    for key in [
        "status",
        "message",
        "current_seed",
        "current_run_name",
        "current_phase",
        "current_attempt",
        "current_num_envs",
        "summary_path",
        "updated_at",
    ]:
        print(f"- {key}: {state.get(key)}")
    print(f"- runner_pid: {pid}")
    print(f"- runner_pid_running: {running}")
else:
    raise SystemExit(f"不支持的子命令: {cmd}")
PY
}

python_autoresearch_state() {
python3 - "$AUTORESEARCH_STATE_PATH" "$AUTORESEARCH_PID_FILE" "$@" <<'PY'
import json
import pathlib
import subprocess
import sys

state_path = pathlib.Path(sys.argv[1])
pid_path = pathlib.Path(sys.argv[2])
cmd = sys.argv[3] if len(sys.argv) > 3 else "status"

def load_state():
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))

def runner_pid():
    if not pid_path.exists():
        return None, False
    value = pid_path.read_text(encoding="utf-8").strip()
    if not value:
        return None, False
    try:
        pid = int(value)
    except ValueError:
        return value, False
    running = subprocess.run(["bash", "-lc", f"ps -p {pid} >/dev/null 2>&1"], check=False).returncode == 0
    return pid, running

state = load_state()
pid, running = runner_pid()

if cmd == "status":
    for key in [
        "supervisor_status",
        "message",
        "iteration_index",
        "next_action",
        "next_trial",
        "resume_from",
        "best_score",
        "best_commit",
        "desired_state",
        "pause_scope",
        "last_heartbeat_at",
        "updated_at",
    ]:
        print(f"{key}: {state.get(key)}")
    print(f"active_process_count: {state.get('active_process_count')}")
    print(f"active_child_pid: {state.get('active_child_pid')}")
    print(f"runner_pid: {pid}")
    print(f"runner_pid_running: {running}")
    print("paths:")
    print(f"  state: {state_path}")
    print(f"  events: {state_path.parent / 'events.jsonl'}")
    print(f"  nohup_log: {state_path.parent / 'autoresearch_supervisor.nohup.log'}")
    print(f"  best_candidate: {state_path.parent / 'best_candidate.json'}")
elif cmd == "report":
    print("# DashGo Autoresearch 状态")
    for key in [
        "supervisor_status",
        "message",
        "iteration_index",
        "best_score",
        "next_trial",
        "resume_from",
        "desired_state",
        "last_heartbeat_at",
    ]:
        print(f"- {key}: {state.get(key)}")
    print(f"- runner_pid: {pid}")
    print(f"- runner_pid_running: {running}")
else:
    raise SystemExit(f"不支持的子命令: {cmd}")
PY
}

python_autoresearch_keepalive_state() {
python3 - "$AUTORESEARCH_KEEPALIVE_STATE_PATH" "$AUTORESEARCH_KEEPALIVE_PID_FILE" <<'PY'
import json
import pathlib
import subprocess
import sys

state_path = pathlib.Path(sys.argv[1])
pid_path = pathlib.Path(sys.argv[2])
state = {}
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))

pid = None
running = False
if pid_path.exists():
    raw = pid_path.read_text(encoding="utf-8").strip()
    if raw.isdigit():
        pid = int(raw)
        running = subprocess.run(["bash", "-lc", f"ps -p {pid} >/dev/null 2>&1"], check=False).returncode == 0

for key in [
    "status",
    "started_at",
    "deadline_at",
    "last_check_at",
    "restart_count",
    "last_restart_at",
    "autoresearch_supervisor_pid",
    "autoresearch_supervisor_running",
    "autoresearch_supervisor_status",
    "autoresearch_message",
]:
    print(f"{key}: {state.get(key)}")
print(f"keepalive_pid: {pid}")
print(f"keepalive_pid_running: {running}")
print("paths:")
print(f"  state: {state_path}")
print(f"  events: {state_path.parent / 'keepalive_events.jsonl'}")
print(f"  nohup_log: {state_path.parent / 'autoresearch_keepalive.nohup.log'}")
PY
}

start_regression() {
  local extra_args=("$@")
  local regression_cmd=("${REGRESSION_CMD[@]}")
  local cmd_string
  local pid
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    regression_cmd+=("${extra_args[@]}")
  fi
  if [[ -f "$REGRESSION_PID_FILE" ]]; then
    pid="$(cat "$REGRESSION_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "正式回归已在运行，PID=$pid" >&2
      return 1
    fi
    rm -f "$REGRESSION_PID_FILE"
  fi
  printf -v cmd_string '%q ' "${regression_cmd[@]}"
  {
    printf '[%s] CMD: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$cmd_string"
  } >>"$REGRESSION_NOHUP_LOG"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -lc "cd '$PROJECT_ROOT' && exec $cmd_string" </dev/null >>"$REGRESSION_NOHUP_LOG" 2>&1 &
  else
    nohup bash -lc "cd '$PROJECT_ROOT' && exec $cmd_string" </dev/null >>"$REGRESSION_NOHUP_LOG" 2>&1 &
  fi
  pid=$!
  echo "$pid" >"$REGRESSION_PID_FILE"
  sleep 1
  if ! ps -p "$pid" >/dev/null 2>&1; then
    echo "正式回归启动失败；后台进程未存活。" >&2
    tail -n 40 "$REGRESSION_NOHUP_LOG" >&2 || true
    rm -f "$REGRESSION_PID_FILE"
    return 1
  fi
}

start_autoresearch() {
  local extra_args=("$@")
  local autoresearch_cmd=("${AUTORESEARCH_CMD[@]}")
  local cmd_string
  local pid
  mkdir -p "$AUTORESEARCH_ROOT"
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    autoresearch_cmd+=("${extra_args[@]}")
  fi
  if [[ -f "$AUTORESEARCH_PID_FILE" ]]; then
    pid="$(cat "$AUTORESEARCH_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "autoresearch supervisor 已在运行，PID=$pid"
      return 0
    fi
    rm -f "$AUTORESEARCH_PID_FILE"
  fi
  printf -v cmd_string '%q ' "${autoresearch_cmd[@]}"
  {
    printf '[%s] CMD: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$cmd_string"
  } >>"$AUTORESEARCH_NOHUP_LOG"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -lc "cd '$PROJECT_ROOT' && exec $cmd_string" </dev/null >>"$AUTORESEARCH_NOHUP_LOG" 2>&1 &
  else
    nohup bash -lc "cd '$PROJECT_ROOT' && exec $cmd_string" </dev/null >>"$AUTORESEARCH_NOHUP_LOG" 2>&1 &
  fi
  pid=$!
  echo "$pid" >"$AUTORESEARCH_PID_FILE"
  sleep 1
  if ! ps -p "$pid" >/dev/null 2>&1; then
    echo "autoresearch supervisor 启动失败；后台进程未存活。" >&2
    tail -n 60 "$AUTORESEARCH_NOHUP_LOG" >&2 || true
    rm -f "$AUTORESEARCH_PID_FILE"
    return 1
  fi
}

start_autoresearch_keepalive() {
  local hours="${1:-4}"
  local keepalive_cmd=("${AUTORESEARCH_KEEPALIVE_CMD[@]}" --hours "$hours")
  local cmd_string
  local pid
  mkdir -p "$AUTORESEARCH_ROOT"
  if [[ -f "$AUTORESEARCH_KEEPALIVE_PID_FILE" ]]; then
    pid="$(cat "$AUTORESEARCH_KEEPALIVE_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "autoresearch keepalive 已在运行，PID=$pid"
      return 0
    fi
    rm -f "$AUTORESEARCH_KEEPALIVE_PID_FILE"
  fi
  printf -v cmd_string '%q ' "${keepalive_cmd[@]}"
  {
    printf '[%s] CMD: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$cmd_string"
  } >>"$AUTORESEARCH_KEEPALIVE_NOHUP_LOG"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -lc "cd '$PROJECT_ROOT' && exec $cmd_string" </dev/null >>"$AUTORESEARCH_KEEPALIVE_NOHUP_LOG" 2>&1 &
  else
    nohup bash -lc "cd '$PROJECT_ROOT' && exec $cmd_string" </dev/null >>"$AUTORESEARCH_KEEPALIVE_NOHUP_LOG" 2>&1 &
  fi
  pid=$!
  echo "$pid" >"$AUTORESEARCH_KEEPALIVE_PID_FILE"
  sleep 1
  if ! ps -p "$pid" >/dev/null 2>&1; then
    echo "autoresearch keepalive 启动失败；后台进程未存活。" >&2
    tail -n 60 "$AUTORESEARCH_KEEPALIVE_NOHUP_LOG" >&2 || true
    rm -f "$AUTORESEARCH_KEEPALIVE_PID_FILE"
    return 1
  fi
}

terminate_pid_tree() {
  local pid="$1"
  local label="$2"
  local grace_sec="${3:-5}"
  local i
  if [[ -z "${pid:-}" ]]; then
    return 0
  fi
  if ps -p "$pid" >/dev/null 2>&1; then
    kill -- "-$pid" >/dev/null 2>&1 || true
    kill "$pid" >/dev/null 2>&1 || true
    for ((i=0; i<grace_sec; i++)); do
      if ! ps -p "$pid" >/dev/null 2>&1; then
        return 0
      fi
      sleep 1
    done
    kill -KILL -- "-$pid" >/dev/null 2>&1 || true
    kill -KILL "$pid" >/dev/null 2>&1 || true
    sleep 1
  fi
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "警告: ${label} 仍未退出，PID=$pid。" >&2
    return 1
  fi
  return 0
}

set_json_state_field() {
  local target_path="$1"
  local desired_state="$2"
  local pause_scope="${3:-all}"
  python3 - "$target_path" "$desired_state" "$pause_scope" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
desired_state = sys.argv[2]
pause_scope = sys.argv[3]
state = {}
if path.exists():
    state = json.loads(path.read_text(encoding="utf-8"))
state["desired_state"] = desired_state
state["pause_scope"] = None if pause_scope == "none" else pause_scope
state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
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

cleanup_autoresearch_state() {
  local target_path="$1"
  local supervisor_status="$2"
  local message="$3"
  python3 - "$target_path" "$supervisor_status" "$message" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
supervisor_status = sys.argv[2]
message = sys.argv[3]
state = {}
if path.exists():
    state = json.loads(path.read_text(encoding="utf-8"))
state["supervisor_status"] = supervisor_status
state["message"] = message
state["desired_state"] = None
state["pause_scope"] = None
state["active_process_count"] = 0
state["active_process_count_real"] = 0
state["active_child_pid"] = None
state["next_action"] = None
state["next_trial"] = None
state["codex_job"] = None
state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
}

cleanup_keepalive_state() {
  local target_path="$1"
  local status="$2"
  python3 - "$target_path" "$status" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
status = sys.argv[2]
state = {}
if path.exists():
    state = json.loads(path.read_text(encoding="utf-8"))
state["status"] = status
state["autoresearch_message"] = None
state["autoresearch_supervisor_pid"] = None
state["autoresearch_supervisor_running"] = False
state["autoresearch_supervisor_status"] = "stopped"
state["keepalive_pid"] = None
state["keepalive_pid_running"] = False
state["cleaned_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
path.parent.mkdir(parents=True, exist_ok=True)
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
  regression-start)
    start_regression "$@"
    echo "已后台启动正式回归。"
    ;;
  regression-status)
    require_file "$REGRESSION_STATE_PATH"
    python_regression_state status
    ;;
  regression-watch)
    interval="${1:-5}"
    while true; do
      clear
      if [[ -f "$REGRESSION_STATE_PATH" ]]; then
        python_regression_state status
      else
        echo "当前还没有 regression_state.json"
      fi
      echo
      echo "---- regression 日志尾部 ----"
      tail -n 20 "$REGRESSION_NOHUP_LOG" 2>/dev/null || true
      sleep "$interval"
    done
    ;;
  regression-logs)
    follow="${1:-}"
    if [[ "$follow" == "-f" ]]; then
      tail -f "$REGRESSION_NOHUP_LOG"
    else
      tail -n 120 "$REGRESSION_NOHUP_LOG"
    fi
    ;;
  regression-stop)
    if [[ -f "$REGRESSION_PID_FILE" ]]; then
      pid="$(cat "$REGRESSION_PID_FILE" 2>/dev/null || true)"
      if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
        kill "$pid"
        echo "已发送停止信号到正式回归进程 PID=$pid。"
        rm -f "$REGRESSION_PID_FILE"
      else
        echo "正式回归进程未运行。"
        rm -f "$REGRESSION_PID_FILE"
      fi
    else
      echo "没有 regression PID 文件。"
    fi
    ;;
  regression-resume)
    start_regression --resume-from-state "$@"
    echo "已按 regression_state.json 继续后台回归。"
    ;;
  autoresearch-start)
    start_autoresearch "$@"
    echo "已后台启动 autoresearch supervisor。"
    ;;
  autoresearch-status)
    require_file "$AUTORESEARCH_STATE_PATH"
    python_autoresearch_state status
    ;;
  autoresearch-watch)
    interval="${1:-5}"
    while true; do
      clear
      if [[ -f "$AUTORESEARCH_STATE_PATH" ]]; then
        python_autoresearch_state status
      else
        echo "当前还没有 autoresearch state.json"
      fi
      echo
      echo "---- autoresearch 日志尾部 ----"
      tail -n 20 "$AUTORESEARCH_NOHUP_LOG" 2>/dev/null || true
      sleep "$interval"
    done
    ;;
  autoresearch-logs)
    follow="${1:-}"
    if [[ "$follow" == "-f" ]]; then
      tail -f "$AUTORESEARCH_NOHUP_LOG"
    else
      tail -n 120 "$AUTORESEARCH_NOHUP_LOG"
    fi
    ;;
  autoresearch-pause)
    set_json_state_field "$AUTORESEARCH_STATE_PATH" "pause_after_current_run" "all" >/dev/null
    echo "已写入 autoresearch safe pause 请求。当前研究轮结束后会暂停。"
    ;;
  autoresearch-resume)
    set_json_state_field "$AUTORESEARCH_STATE_PATH" "running" "none" >/dev/null
    start_autoresearch "$@"
    echo "已解除 autoresearch safe pause，并确保 supervisor 运行。"
    ;;
  autoresearch-stop)
    bash "$0" autoresearch-ensure-stop >/dev/null 2>&1 || true
    if [[ -f "$AUTORESEARCH_PID_FILE" ]]; then
      pid="$(cat "$AUTORESEARCH_PID_FILE" 2>/dev/null || true)"
      if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
        terminate_pid_tree "$pid" "autoresearch supervisor" 5
        echo "已停止 autoresearch supervisor，PID=$pid。"
        rm -f "$AUTORESEARCH_PID_FILE"
      else
        echo "autoresearch supervisor 未运行。"
        rm -f "$AUTORESEARCH_PID_FILE"
      fi
    else
      echo "没有 autoresearch PID 文件。"
    fi
    cleanup_autoresearch_state "$AUTORESEARCH_STATE_PATH" "stopped" "已执行彻底清理，autoresearch supervisor 已停止" >/dev/null
    ;;
  autoresearch-report)
    require_file "$AUTORESEARCH_STATE_PATH"
    python_autoresearch_state report
    ;;
  autoresearch-ensure-start)
    hours="${1:-4}"
    start_autoresearch_keepalive "$hours"
    echo "已启动 autoresearch keepalive，最短守护时长 ${hours} 小时。"
    ;;
  autoresearch-ensure-status)
    require_file "$AUTORESEARCH_KEEPALIVE_STATE_PATH"
    python_autoresearch_keepalive_state
    ;;
  autoresearch-ensure-logs)
    follow="${1:-}"
    if [[ "$follow" == "-f" ]]; then
      tail -f "$AUTORESEARCH_KEEPALIVE_NOHUP_LOG"
    else
      tail -n 120 "$AUTORESEARCH_KEEPALIVE_NOHUP_LOG"
    fi
    ;;
  autoresearch-ensure-stop)
    if [[ -f "$AUTORESEARCH_KEEPALIVE_PID_FILE" ]]; then
      pid="$(cat "$AUTORESEARCH_KEEPALIVE_PID_FILE" 2>/dev/null || true)"
      if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
        terminate_pid_tree "$pid" "autoresearch keepalive" 3
        echo "已停止 autoresearch keepalive，PID=$pid。"
        rm -f "$AUTORESEARCH_KEEPALIVE_PID_FILE"
      else
        echo "autoresearch keepalive 未运行。"
        rm -f "$AUTORESEARCH_KEEPALIVE_PID_FILE"
      fi
    else
      echo "没有 autoresearch keepalive PID 文件。"
    fi
    cleanup_keepalive_state "$AUTORESEARCH_KEEPALIVE_STATE_PATH" "stopped" >/dev/null
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
