#!/bin/bash
# =============================================================================
# DashGo Auto-Training Launcher
# 功能：启动训练、监控日志、自动分析、循环优化
# 用法：./auto_train_launcher.sh
# =============================================================================

set -e  # 遇到错误立即退出

# 配置
PROJECT_DIR="/home/gwh/dashgo_rl_project"
ISAACLAB_DIR="/home/gwh/IsaacLab"
TRAIN_SCRIPT="train_v2.py"
LOG_DIR="$PROJECT_DIR/logs"
ISSUE_DIR="$PROJECT_DIR/issues"
MONITOR_INTERVAL=60  # 监控间隔（秒）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  DashGo Auto-Training System v1.0${NC}"
echo -e "${GREEN}========================================${NC}"

# =============================================================================
# Phase 1: 启动训练
# =============================================================================

echo -e "\n${YELLOW}[Phase 1] 启动训练...${NC}"

# 清理旧日志（可选，如果想保留历史则注释掉）
# rm -rf "$LOG_DIR"/*

# 进入项目目录
cd "$PROJECT_DIR"

# 启动训练（后台运行）
nohup "$ISAACLAB_DIR/isaaclab.sh" \
    -p "$TRAIN_SCRIPT" \
    --headless \
    --num_envs 4096 \
    > training_output.log 2>&1 &

TRAIN_PID=$!
echo "训练进程PID: $TRAIN_PID"
echo "训练输出: $PROJECT_DIR/training_output.log"

# 等待训练启动
sleep 10

# 检查进程是否还在运行
if ps -p $TRAIN_PID > /dev/null; then
    echo -e "${GREEN}✓ 训练成功启动 (PID: $TRAIN_PID)${NC}"
else
    echo -e "${RED}✗ 训练启动失败，请检查日志${NC}"
    exit 1
fi

# =============================================================================
# Phase 2: 实时监控
# =============================================================================

echo -e "\n${YELLOW}[Phase 2] 启动实时监控...${NC}"

# 监控函数
monitor_training() {
    local iteration=0
    local max_reach_goal=0
    local max_collision=100

    echo -e "${GREEN}开始监控训练进度（每 ${MONITOR_INTERVAL}s 更新一次）...${NC}"
    echo -e "${GREEN}按 Ctrl+C 停止监控（不影响训练）${NC}\n"

    # 创建监控日志
    MONITOR_LOG="$ISSUE_DIR/monitoring_$(date +%Y%m%d_%H%M%S).log"
    echo "监控日志: $MONITOR_LOG"

    while ps -p $TRAIN_PID > /dev/null; do
        # 解析TensorBoard日志
        if ls "$LOG_DIR"/events.out.tfevents.* > /dev/null 2>&1; then
            # 获取最新的event文件
            LATEST_EVENT=$(ls -t "$LOG_DIR"/events.out.tfevents.* 2>/dev/null | head -1)

            # 使用tensorboard --logdir读取数据（如果安装了）
            # 或者直接解析日志文件
            iteration=$(grep -o "iteration [0-9]*" training_output.log 2>/dev/null | tail -1 | grep -o "[0-9]*" || echo "N/A")

            # 简化的监控（从训练输出解析）
            if [ -f training_output.log ]; then
                # 提取关键指标
                current_iter=$(tail -100 training_output.log | grep -o "Iteration [0-9]*" | tail -1 | grep -o "[0-9]*" || echo "0")

                # 提取reach_goal率（如果有的话）
                reach_goal_rate=$(tail -100 training_output.log | grep -o "reach_goal [0-9.]*%*" | tail -1 || echo "N/A")

                # 提取Policy Noise
                policy_noise=$(tail -100 training_output.log | grep -o "action noise std: [0-9.]*" | tail -1 | grep -o "[0-9.]*" || echo "N/A")

                echo -e "${NC}[$iteration] Iteration: $current_iter | reach_goal: $reach_goal_rate | Noise: $policy_noise"

                # 记录到监控日志
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iteration: $current_iter | reach_goal: $reach_goal_rate | Noise: $policy_noise" >> "$MONITOR_LOG"
            fi
        fi

        sleep $MONITOR_INTERVAL
    done

    echo -e "${YELLOW}训练进程已结束${NC}"
}

# 启动监控（后台运行）
monitor_training &
MONITOR_PID=$!

# =============================================================================
# Phase 3: 等待训练结束
# =============================================================================

echo -e "\n${YELLOW}[Phase 3] 等待训练完成...${NC}"
echo "训练PID: $TRAIN_PID"
echo "监控PID: $MONITOR_PID"
echo ""

# 等待训练进程结束
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# 停止监控
kill $MONITOR_PID 2>/dev/null || true

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ 训练正常完成${NC}"
else
    echo -e "${RED}✗ 训练异常退出 (退出码: $TRAIN_EXIT_CODE)${NC}"
fi

# =============================================================================
# Phase 4: 分析结果
# =============================================================================

echo -e "\n${YELLOW}[Phase 4] 分析训练结果...${NC}"

# 生成分析报告
ANALYSIS_SCRIPT="$PROJECT_DIR/auto_analyze.py"

if [ -f "$ANALYSIS_SCRIPT" ]; then
    python3 "$ANALYSIS_SCRIPT" "auto"
else
    echo -e "${YELLOW}⚠ 未找到分析脚本，跳过自动分析${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Auto-Training 流程完成${NC}"
echo -e "${GREEN}========================================${NC}"
