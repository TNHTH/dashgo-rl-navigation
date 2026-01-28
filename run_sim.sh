#!/bin/bash
# DashGo RL导航 - 仿真模式启动脚本
#
# 用途：一键启动Gazebo仿真和RL导航节点
# 使用：./run_sim.sh

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DashGo RL 导航 - 仿真模式${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}[WARN] 未激活conda环境，正在激活dashgo_deploy...${NC}"
    conda activate dashgo_deploy
fi

echo -e "${GREEN}[INFO] 当前环境: $CONDA_DEFAULT_ENV${NC}"

# 加载ROS工作空间
if [ -f "catkin_ws/devel/setup.bash" ]; then
    source catkin_ws/devel/setup.bash
    echo -e "${GREEN}[INFO] ROS工作空间已加载${NC}"
else
    echo -e "${YELLOW}[WARN] 未找到devel/setup.bash，尝试使用source...${NC}"
    source catkin_ws/devel/setup.bash 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}[INFO] 启动Gazebo仿真和RL节点...${NC}"
echo ""
echo "提示："
echo "  - Gazebo窗口将自动打开"
echo "  - Rviz将自动打开"
echo "  - 使用Rviz的'2D Nav Goal'工具发送目标点"
echo ""

# 启动仿真launch文件
roslaunch dashgo_rl sim2real_golden.launch
