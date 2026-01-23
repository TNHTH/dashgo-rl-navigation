#!/bin/bash

echo "[INFO] 正在强制清理残留的 Isaac Sim 进程..."
# 强制杀死所有包含 'kit' 或 'python train_v2.py' 的进程
pkill -9 -f "kit"
pkill -9 -f "python train_v2.py"
sleep 2

echo "[INFO] 显存清理完毕。"
echo "[INFO] 正在以【无头模式 + 低负载】启动训练..."
echo "[NOTE] 无头模式下不会显示图形界面，但训练速度更快且不易崩溃。"

# 关键参数解释：
# --headless: 关闭图形界面，节省约 2-3GB 显存给相机使用
# --num_envs 16: 将环境数降至 16，确保 4060 Laptop 能跑通
# --enable_cameras: 开启相机渲染 (这是我们模拟雷达必须的)

python train_v2.py --headless --num_envs 16 --enable_cameras

# 如果您想续训，请取消下面这行的注释：
# python train_v2.py --headless --num_envs 16 --enable_cameras --resume