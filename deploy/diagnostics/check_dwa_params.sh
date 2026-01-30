#!/bin/bash
# 检查DWA局部规划器参数

echo "=== DWA参数检查 ==="
echo ""

# 检查DWA是否启用
echo "【1. DWA规划器状态】"
if rosnode list | grep -q "DWAPlannerROS"; then
    echo "✅ DWA规划器运行中"
else
    echo "⚠️  DWA规划器节点未找到"
    echo "   检查move_base配置..."
fi
echo ""

# 检查速度限制
echo "【2. 速度限制】"
echo "最大线速度 (max_vel_x):"
rosparam get /move_base/DWAPlannerROS/max_vel_x 2>/dev/null || echo "  ❌ 无法获取"

echo "最小线速度 (min_vel_x):"
rosparam get /move_base/DWAPlannerROS/min_vel_x 2>/dev/null || echo "  ❌ 无法获取"

echo "最大角速度 (max_rot_vel):"
rosparam get /move_base/DWAPlannerROS/max_rot_vel 2>/dev/null || echo "  ❌ 无法获取"
echo ""

# 检查加速度限制
echo "【3. 加速度限制】"
echo "线加速度 (acc_lim_x):"
rosparam get /move_base/DWAPlannerROS/acc_lim_x 2>/dev/null || echo "  ❌ 无法获取"

echo "角加速度 (acc_lim_theta):"
rosparam get /move_base/DWAPlannerROS/acc_lim_theta 2>/dev/null || echo "  ❌ 无法获取"
echo ""

# 检查路径跟随权重
echo "【4. 路径跟随权重】"
echo "路径距离权重 (path_distance_bias):"
rosparam get /move_base/DWAPlannerROS/path_distance_bias 2>/dev/null || echo "  ❌ 无法获取"

echo "目标距离权重 (goal_distance_bias):"
rosparam get /move_base/DWAPlannerROS/goal_distance_bias 2>/dev/null || echo "  ❌ 无法获取"

echo "_occdist_scale (障碍物避让权重):"
rosparam get /move_base/DWAPlannerROS/occdist_scale 2>/dev/null || echo "  ❌ 无法获取"
echo ""

# 检查机器人尺寸
echo "【5. 机器人尺寸（代价地图）】"
echo "机器人半径 (robot_radius):"
rosparam get /move_base/global_costmap/robot_radius 2>/dev/null || echo "  ❌ 无法获取"

echo "障碍物膨胀半径 (inflation_radius):"
rosparam get /move_base/global_costmap/inflation_radius 2>/dev/null || echo "  ❌ 无法获取"
echo ""

# 安全性评估
echo "【6. 安全性评估】"
MAX_VEL=$(rosparam get /move_base/DWAPlannerROS/max_vel_x 2>/dev/null)
if [ ! -z "$MAX_VEL" ]; then
    if (( $(echo "$MAX_VEL > 0.3" | bc -l) )); then
        echo "⚠️  最大线速度过高 (${MAX_VEL} m/s > 0.3 m/s)"
        echo "   建议：降低到0.2 m/s以下"
    else
        echo "✅ 最大线速度合理 (${MAX_VEL} m/s)"
    fi
fi

INF_RADIUS=$(rosparam get /move_base/global_costmap/inflation_radius 2>/dev/null)
if [ ! -z "$INF_RADIUS" ]; then
    if (( $(echo "$INF_RADIUS < 0.3" | bc -l) )); then
        echo "⚠️  障碍物膨胀半径过小 (${INF_RADIUS} m < 0.3 m)"
        echo "   建议：增加到0.4-0.5 m"
    else
        echo "✅ 障碍物膨胀半径合理 (${INF_RADIUS} m)"
    fi
fi

echo ""
echo "=== 检查完成 ==="
