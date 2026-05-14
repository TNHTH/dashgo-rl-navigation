#!/usr/bin/env bash
set -euo pipefail

# DashGo 当前避障/导航能力检查入口
# 默认使用当前更稳妥的 Gen2 checkpoint:
#   wave44/model_704.pt
# 如需验证最新有限波次结果，可传:
#   --checkpoint latest

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}"
AUTOPILOT_ROOT="$PROJECT_ROOT/.artifacts/autopilot"
PLAY_SCRIPT="$PROJECT_ROOT/apps/isaac/play.py"
LIVE_ENV_SCRIPT="$PROJECT_ROOT/tools/diagnostics/inspect_live_env.py"
VERIFY_SCRIPT="$PROJECT_ROOT/apps/isaac/verify_ultimate_v5.py"

STABLE_CHECKPOINT="$AUTOPILOT_ROOT/runs/gen2/20260319_024028_wave44_gen2_model655_stablehistory_seed44_capture/checkpoints/model_704.pt"
LATEST_CHECKPOINT="$AUTOPILOT_ROOT/runs/gen2/20260319_024319_wave45_gen2_model704_stablehistory_extend375/checkpoints/model_743.pt"

mode="play"
profile="gen2"
checkpoint_input="stable"
num_envs="1"
steps="12"
goal_x="3.0"
goal_y="0.0"
goal_z="0.0"
headless="0"
extra_args=()

usage() {
  cat <<EOF
用法:
  $(basename "$0") [play|sensor|fullstack] [选项] [-- 额外参数]

模式:
  play        GUI/Headless 回放当前模型，直接看避障与导航行为
  sensor      运行训练环境活体诊断，检查传感器/奖励链是否健康
  fullstack   运行 verify_ultimate_v5.py，全栈健康检查

常用选项:
  --checkpoint stable|latest|/abs/path/model_xxx.pt
  --profile gen2|gen1
  --num-envs N
  --steps N              仅 sensor 模式使用
  --goal-x X             仅 play 模式使用
  --goal-y Y             仅 play 模式使用
  --goal-z Z             仅 play 模式使用
  --headless             仅 play 模式使用
  -h, --help

默认 checkpoint:
  stable -> $STABLE_CHECKPOINT
  latest -> $LATEST_CHECKPOINT

示例:
  $(basename "$0")
  $(basename "$0") play --goal-x 3.5 --goal-y 1.5
  $(basename "$0") play --checkpoint latest
  $(basename "$0") sensor --num-envs 4 --steps 16
  $(basename "$0") fullstack
EOF
}

resolve_checkpoint() {
  case "$1" in
    stable)
      printf '%s\n' "$STABLE_CHECKPOINT"
      ;;
    latest)
      printf '%s\n' "$LATEST_CHECKPOINT"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    play|sensor|fullstack)
      mode="$1"
      shift
      ;;
    --checkpoint)
      checkpoint_input="$2"
      shift 2
      ;;
    --profile)
      profile="$2"
      shift 2
      ;;
    --num-envs)
      num_envs="$2"
      shift 2
      ;;
    --steps)
      steps="$2"
      shift 2
      ;;
    --goal-x)
      goal_x="$2"
      shift 2
      ;;
    --goal-y)
      goal_y="$2"
      shift 2
      ;;
    --goal-z)
      goal_z="$2"
      shift 2
      ;;
    --headless)
      headless="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$ISAACLAB_SH" ]]; then
  echo "❌ 未找到可执行 Isaac Lab 启动脚本: $ISAACLAB_SH" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export DASHGO_AUTOPILOT_PROFILE="$profile"

case "$mode" in
  play)
    checkpoint="$(resolve_checkpoint "$checkpoint_input")"
    if [[ ! -f "$checkpoint" ]]; then
      echo "❌ checkpoint 不存在: $checkpoint" >&2
      exit 1
    fi

    echo "== DashGo 能力回放 =="
    echo "profile    : $profile"
    echo "checkpoint : $checkpoint"
    echo "goal       : ($goal_x, $goal_y, $goal_z)"
    echo "num_envs   : $num_envs"
    echo
    echo "观察重点:"
    echo "1. 是否主动绕开近障，而不是直冲或前蹭"
    echo "2. 是否持续朝 waypoint / goal 推进，而不是原地磨蹭"
    echo "3. 是否出现频繁倒车后卡死或长期 timeout"
    echo

    cmd=(
      "$ISAACLAB_SH" -p "$PLAY_SCRIPT"
      --num_envs "$num_envs"
      --checkpoint "$checkpoint"
      --goal_x "$goal_x"
      --goal_y "$goal_y"
      --goal_z "$goal_z"
    )
    if [[ "$headless" == "1" ]]; then
      cmd+=(--headless)
    fi
    cmd+=("${extra_args[@]}")
    exec "${cmd[@]}"
    ;;

  sensor)
    echo "== DashGo 训练环境活体诊断 =="
    echo "profile  : $profile"
    echo "num_envs : $num_envs"
    echo "steps    : $steps"
    echo

    cmd=(
      "$ISAACLAB_SH" -p "$LIVE_ENV_SCRIPT"
      --headless
      --profile "$profile"
      --num_envs "$num_envs"
      --steps "$steps"
    )
    cmd+=("${extra_args[@]}")
    exec "${cmd[@]}"
    ;;

  fullstack)
    echo "== DashGo 全栈健康检查 =="
    echo "profile: $profile"
    echo
    cmd=("$ISAACLAB_SH" -p "$VERIFY_SCRIPT")
    cmd+=("${extra_args[@]}")
    exec "${cmd[@]}"
    ;;
esac
