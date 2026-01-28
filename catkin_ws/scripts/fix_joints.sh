#!/bin/bash
# ROS环境清洗脚本 - 解决"幽灵环境"问题

echo "🔧 正在清洗Python环境..."

# 1. 彻底清空Conda环境变量
unset PYTHONPATH
unset PYTHONHOME
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PROMPT_MODIFIER

# 2. 重新加载ROS环境（干净的）
source /opt/ros/noetic/setup.bash
source ~/dashgo_rl_project/catkin_ws/devel/setup.bash

# 3. 验证Python路径
echo "✅ Python路径: $(which python3)"
echo "✅ yaml模块: $(/usr/bin/python3 -c 'import yaml; print(yaml.__version__)')"

# 4. 启动ROS节点
echo "🚀 启动ROS节点..."
exec roslaunch dashgo_rl sim2real_golden.launch
