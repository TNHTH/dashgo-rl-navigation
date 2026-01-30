#!/bin/bash
# 阶段MVP-1：环境诊断脚本
# 用途：验证ROS环境，确认plan topic名称

echo "========================================"
echo "阶段MVP-1：环境诊断"
echo "========================================"
echo ""

# 检查是否在项目目录
if [ ! -d "catkin_ws" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    echo "   当前目录: $(pwd)"
    echo "   应该在: /home/gwh/dashgo_rl_project"
    exit 1
fi

echo "✅ 项目目录确认"
echo ""

# 加载ROS环境
echo "正在加载ROS环境..."
source catkin_ws/devel/setup.bash

echo "========================================"
echo "步骤1.2：检查ROS节点和话题"
echo "========================================"
echo ""

# 检查move_base节点
echo "1️⃣ 检查move_base节点..."
MOVE_BASE_NODES=$(rosnode list 2>/dev/null | grep move_base)
if [ -n "$MOVE_BASE_NODES" ]; then
    echo "✅ 找到move_base节点:"
    echo "$MOVE_BASE_NODES"
else
    echo "❌ 未找到move_base节点"
    echo "   请确认是否启动了: roslaunch dashgo_rl sim2real_golden.launch"
fi
echo ""

# 检查plan话题
echo "2️⃣ 检查plan话题（关键！）..."
PLAN_TOPICS=$(rostopic list 2>/dev/null | grep -E "plan|Navfn|Global")
if [ -n "$PLAN_TOPICS" ]; then
    echo "✅ 找到plan相关话题:"
    echo "$PLAN_TOPICS"
    echo ""
    echo "🔍 请记录这些话题名称，将用于代码修改"
else
    echo "❌ 未找到plan相关话题"
    echo "   这可能意味着："
    echo "   - move_base未正确配置"
    echo "   - 或者需要先发送一个目标点"
fi
echo ""

# 检查TF树
echo "3️⃣ 检查TF树（map → base_link）..."
TF_RESULT=$(timeout 2 rosrun tf tf_echo map base_link 2>&1)
if echo "$TF_RESULT" | grep -q "Transform"; then
    echo "✅ TF树正常"
else
    echo "⚠️ TF树可能有问题"
fi
echo ""

echo "========================================"
echo "步骤1.3：测试目标点发送"
echo "========================================"
echo ""
echo "📝 请在另一个终端执行以下命令："
echo ""
echo "   # 发布测试目标点（2米外）"
echo "   rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \\"
echo "     '{header: {frame_id: \"map\"}, pose: {position: {x: 2.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}' \\"
echo "     --once"
echo ""
echo "   # 然后立即观察plan话题"
echo "   rostopic echo /move_base/GlobalPlanner/plan -n 1"
echo ""
echo "   # 或者根据上面显示的实际topic名称"
echo ""

echo "========================================"
echo "诊断完成"
echo "========================================"
echo ""
echo "📋 请记录以下信息："
echo ""
echo "1. plan话题的完整名称:"
echo "   _______________________________"
echo ""
echo "2. 路径点数量范围:"
echo "   最少_____ 最多_____"
echo ""
echo "3. 错误信息（如果有）:"
echo "   _______________________________"
echo ""
echo "✅ 诊断脚本执行完成"
