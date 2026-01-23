#!/bin/bash
#
# DashGo RL Headless 训练启动脚本
#
# 使用方法:
#   ./run_headless_train.sh --num_envs 80
#

# 设置 Isaac Sim 为纯 headless 模式（不依赖 X server）
export IsaacLAB_WINDOWED=0
export OMNI_KIT_ACCEPT_EULA=YES

# 如果没有显示器，强制使用虚拟显示
if [ -z "$DISPLAY" ]; then
    echo "[INFO] 未检测到 DISPLAY，使用纯 headless 模式"
    export DISPLAY=""
fi

# 启动训练（传递所有参数）
~/IsaacLab/isaaclab.sh -p train_v2.py --headless "$@"
