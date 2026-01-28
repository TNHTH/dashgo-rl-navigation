#!/bin/bash
# DashGo RL导航 - 实车模式启动脚本
#
# 用途：一键连接机器人并启动RL导航节点
# 使用：./run_real.sh [机器人IP]
#
# 示例：
#   ./run_real.sh              # 使用默认IP（交互式输入）
#   ./run_real.sh 192.168.5.100 # 直接指定机器人IP

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DashGo RL 导航 - 实车模式${NC}"
echo -e "${GREEN}========================================${NC}"

# ============================================================================
# 1. 激活conda环境
# ============================================================================
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "dashgo_deploy" ]; then
    echo -e "${YELLOW}[INFO] 激活dashgo_deploy环境...${NC}"
    conda activate dashgo_deploy
fi

echo -e "${GREEN}[INFO] 当前环境: $CONDA_DEFAULT_ENV${NC}"

# ============================================================================
# 2. 加载ROS工作空间
# ============================================================================
if [ -f "catkin_ws/devel/setup.bash" ]; then
    source catkin_ws/devel/setup.bash
    echo -e "${GREEN}[INFO] ROS工作空间已加载${NC}"
else
    echo -e "${RED}[ERROR] 未找到devel/setup.bash，请先编译工作空间！${NC}"
    echo "运行: cd catkin_ws && catkin_make -DPYTHON_EXECUTABLE=\$(which python)"
    exit 1
fi

# ============================================================================
# 3. 连接机器人（使用connect_to_robot.sh脚本）
# ============================================================================
echo ""
echo -e "${GREEN}[INFO] 配置ROS多机连接...${NC}"
bash connect_to_robot.sh $1

# 检查连接是否成功
if ! rostopic list &> /dev/null; then
    echo ""
    echo -e "${RED}[ERROR] 无法连接到机器人，启动中止！${NC}"
    echo "请检查："
    echo "  1. 机器人上位机是否启动roscore"
    echo "  2. 网络是否连通"
    echo "  3. IP地址是否正确"
    exit 1
fi

# ============================================================================
# 4. 启动RL节点
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}[INFO] 启动RL导航节点...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "提示："
echo "  - Rviz将自动打开，显示机器人实时状态"
echo "  - 使用Rviz的'2D Nav Goal'工具发送目标点"
echo "  - 按 Ctrl+C 退出"
echo ""

# 启动实车控制launch文件
roslaunch dashgo_rl real_control.launch
