#!/bin/bash
# 方案C：纯主机+aptitude安装ros-noetic-navigation
# 架构师推荐方案（2026-01-29）
#
# 用法：sudo bash deploy/host/install_navigation_aptitude.sh

set -e  # 遇到错误立即退出

echo "=== 方案C：使用aptitude安装ros-noetic-navigation ==="
echo ""
echo "⚠️  注意：此脚本会降级libsdl1.2库以解决依赖冲突"
echo "⚠️  这是架构师推荐的'微创手术'方案"
echo ""

# 检查是否有sudo权限
if [ "$EUID" -ne 0 ]; then
   echo "❌ 错误：此脚本需要sudo权限"
   echo "请使用: sudo bash $0"
   exit 1
fi

echo "=== 步骤1：安装aptitude ==="
sudo apt update
sudo apt install -y aptitude

echo ""
echo "=== 步骤2：使用aptitude安装navigation ==="
echo "⚠️  接下来会有交互式对话，请仔细阅读："
echo ""
echo "【第一问】"
echo "  系统会问：Accept this solution? [Y/n/q/?]"
echo "  会显示：Keep the following packages at their current version"
echo "  ⚠️ 你必须输入: n  （我们要安装，不是保持现状）"
echo ""
echo "【第二问】"
echo "  系统会尝试降级方案，例如："
echo "  Downgrade libsdl1.2 from version 2.0-0 to 1.2.15"
echo "  ⚠️ 只要没看到'Remove ros-noetic-desktop'，就输入: y"
echo ""

read -p "按Enter继续执行aptitude安装，或Ctrl+C取消..."

# 使用aptitude安装
sudo aptitude install ros-noetic-navigation

echo ""
echo "=== 步骤3：验证安装 ==="
if rospack find move_base >/dev/null 2>&1; then
    echo "✅ move_base安装成功！"
    echo "   路径: $(rospack find move_base)"
else
    echo "❌ move_base安装失败"
    exit 1
fi

echo ""
echo "=== 安装完成！ ==="
echo ""
echo "下一步："
echo "1. 启动Gazebo仿真测试："
echo "   roslaunch dashgo_rl sim2real_golden.launch enable_gazebo:=true enable_move_base:=true"
echo ""
echo "2. 启动RL节点（需要先conda activate env_isaaclab）："
echo "   python catkin_ws/src/dashgo_rl/scripts/geo_nav_node.py"
