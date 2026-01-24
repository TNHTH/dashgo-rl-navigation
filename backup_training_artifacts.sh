#!/bin/bash
################################################################################
# 训练产物自动备份脚本
#
# 功能：
#   1. 将 logs/ 目录下的模型和TensorBoard事件文件移动到备份目录
#   2. 按日期分类组织备份文件
#   3. 清空 logs/ 目录为下次训练准备
#
# 使用：
#   ./backup_training_artifacts.sh
#
# 作者：Claude Code AI Assistant
# 创建时间：2026-01-24
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
BACKUP_ROOT="${PROJECT_ROOT}/logs_backup"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATE_ONLY=$(date +"%Y-%m-%d")

echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}       训练产物自动备份脚本${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo ""
echo "项目目录: ${PROJECT_ROOT}"
echo "备份根目录: ${BACKUP_ROOT}"
echo "时间戳: ${TIMESTAMP}"
echo ""

# 检查 logs 目录是否存在
if [ ! -d "${LOGS_DIR}" ]; then
    echo -e "${YELLOW}⚠️  logs 目录不存在，无需备份${NC}"
    exit 0
fi

# 检查是否有训练产物
MODEL_COUNT=$(find "${LOGS_DIR}" -name "model_*.pt" 2>/dev/null | wc -l)
EVENT_COUNT=$(find "${LOGS_DIR}" -name "events.out.tfevents.*" 2>/dev/null | wc -l)

if [ "${MODEL_COUNT}" -eq 0 ] && [ "${EVENT_COUNT}" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  logs 目录中没有训练产物，无需备份${NC}"
    exit 0
fi

echo -e "${GREEN}发现训练产物：${NC}"
echo "  - 模型文件: ${MODEL_COUNT} 个"
echo "  - TensorBoard事件: ${EVENT_COUNT} 个"
echo ""

# 创建备份目录结构
BACKUP_DIR="${BACKUP_ROOT}/${DATE_ONLY}/${TIMESTAMP}_reward_hacking_fixed"
MODELS_DIR="${BACKUP_DIR}/models"
TENSORBOARD_DIR="${BACKUP_DIR}/tensorboard"
MISC_DIR="${BACKUP_DIR}/misc"

mkdir -p "${MODELS_DIR}"
mkdir -p "${TENSORBOARD_DIR}"
mkdir -p "${MISC_DIR}"

echo -e "${GREEN}创建备份目录结构：${NC}"
echo "  - ${BACKUP_DIR}"
echo "    ├── models/        (模型文件 *.pt)"
echo "    ├── tensorboard/   (TensorBoard事件文件)"
echo "    └── misc/          (其他文件)"
echo ""

# 备份模型文件
if [ "${MODEL_COUNT}" -gt 0 ]; then
    echo -e "${GREEN}📦 备份模型文件...${NC}"
    MODEL_BACKUP_COUNT=0

    while IFS= read -r -d '' model_file; do
        filename=$(basename "${model_file}")
        cp "${model_file}" "${MODELS_DIR}/${filename}"
        MODEL_BACKUP_COUNT=$((MODEL_BACKUP_COUNT + 1))
    done < <(find "${LOGS_DIR}" -maxdepth 1 -name "model_*.pt" -print0 2>/dev/null)

    echo "  ✓ 已备份 ${MODEL_BACKUP_COUNT} 个模型文件到 ${MODELS_DIR}"
fi

# 备份 TensorBoard 事件文件
if [ "${EVENT_COUNT}" -gt 0 ]; then
    echo -e "${GREEN}📊 备份 TensorBoard 事件文件...${NC}"
    EVENT_BACKUP_COUNT=0

    while IFS= read -r -d '' event_file; do
        filename=$(basename "${event_file}")
        cp "${event_file}" "${TENSORBOARD_DIR}/${filename}"
        EVENT_BACKUP_COUNT=$((EVENT_BACKUP_COUNT + 1))
    done < <(find "${LOGS_DIR}" -maxdepth 1 -name "events.out.tfevents.*" -print0 2>/dev/null)

    echo "  ✓ 已备份 ${EVENT_BACKUP_COUNT} 个事件文件到 ${TENSORBOARD_DIR}"
fi

# 备份其他文件（rsl_rl目录等）
echo -e "${GREEN}📁 备份其他文件...${NC}"
MISC_BACKUP_COUNT=0

# 备份 rsl_rl 目录（如果存在）
if [ -d "${LOGS_DIR}/rsl_rl" ]; then
    cp -r "${LOGS_DIR}/rsl_rl" "${MISC_DIR}/"
    MISC_BACKUP_COUNT=$((MISC_BACKUP_COUNT + 1))
    echo "  ✓ 已备份 rsl_rl/ 目录"
fi

# 备份 git 目录（如果存在）
if [ -d "${LOGS_DIR}/git" ]; then
    cp -r "${LOGS_DIR}/git" "${MISC_DIR}/"
    MISC_BACKUP_COUNT=$((MISC_BACKUP_COUNT + 1))
    echo "  ✓ 已备份 git/ 目录"
fi

echo ""
echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}✅ 备份完成！${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo ""
echo "备份位置: ${BACKUP_DIR}"
echo ""
echo "备份统计:"
echo "  - 模型文件: ${MODEL_BACKUP_COUNT} 个"
echo "  - TensorBoard事件: ${EVENT_BACKUP_COUNT} 个"
echo "  - 其他文件: ${MISC_BACKUP_COUNT} 项"
echo ""
echo -e "${YELLOW}接下来：${NC}"
echo "  1. 检查备份是否正确: ls -lh ${BACKUP_DIR}"
echo "  2. 清空 logs 目录准备新训练: rm -rf ${LOGS_DIR}/*"
echo "  3. 启动新训练: ~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 256"
echo ""
