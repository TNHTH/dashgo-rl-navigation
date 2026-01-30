#!/bin/bash
# Sim2Real系统诊断脚本
# 检查SLAM、全局规划、局部规划（RL）三大组件状态

echo "=== Sim2Real系统诊断 ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_node() {
    if rosnode list 2>/dev/null | grep -q "$1"; then
        echo -e "${GREEN}✅${NC} $1 节点运行中"
        return 0
    else
        echo -e "${RED}❌${NC} $1 节点未运行"
        return 1
    fi
}

check_topic() {
    if rostopic list 2>/dev/null | grep -q "$1"; then
        echo -e "${GREEN}✅${NC} $1 话题存在"
        return 0
    else
        echo -e "${RED}❌${NC} $1 话题不存在"
        return 1
    fi
}

check_topic_hz() {
    local topic=$1
    local min_hz=$2
    local hz=$(rostopic hz $topic 2>/dev/null | head -1 | grep -oP 'average: \K[\d.]+')

    if [ ! -z "$hz" ]; then
        if (( $(echo "$hz >= $min_hz" | bc -l) )); then
            echo -e "${GREEN}✅${NC} $topic 频率: ${hz} Hz (正常 >${min_hz}Hz)"
        else
            echo -e "${YELLOW}⚠️${NC}  $topic 频率: ${hz} Hz (偏低，应>${min_hz}Hz)"
        fi
    else
        echo -e "${RED}❌${NC} $topic 无法获取频率"
    fi
}

# === 1. 检查ROS Master ===
echo "【1. ROS Master状态】"
if rostopic list &>/dev/null; then
    echo -e "${GREEN}✅${NC} ROS Master运行中"
    ROS_MASTER_URI=$(echo $ROS_MASTER_URI)
    echo "   ROS_MASTER_URI: $ROS_MASTER_URI"
else
    echo -e "${RED}❌${NC} ROS Master未运行或无法连接"
    echo "   请先启动roscore或launch文件"
    exit 1
fi
echo ""

# === 2. 检查核心节点 ===
echo "【2. 核心节点检查】"
check_node "/gazebo"
check_node "/move_base"
check_node "/slam_gmapping"
check_node "/map_server"
check_node "/geo_nav_node"
echo ""

# === 3. 检查关键话题 ===
echo "【3. 关键话题检查】"
check_topic "/scan"
check_topic "/odom"
check_topic "/cmd_vel"
check_topic "/map"
check_topic "/move_base_simple/goal"
check_topic "/move_base/goal"
check_topic "/move_base/result"
check_topic "/move_base/feedback"
echo ""

# === 4. 检查传感器数据频率 ===
echo "【4. 传感器数据频率检查】"
echo "   (采集3秒数据...)"
check_topic_hz "/scan" 8
check_topic_hz "/odom" 20
check_topic_hz "/cmd_vel" 5
echo ""

# === 5. 检查move_base配置 ===
echo "【5. move_base配置检查】"
BASE_PLANNER=$(rosparam get /move_base/base_local_planner 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅${NC} 局部规划器类型: $BASE_PLANNER"

    if echo "$BASE_PLANNER" | grep -q "dwa"; then
        echo "   → 使用DWA局部规划器（传统方案A）"
    elif echo "$BASE_PLANNER" | grep -q "teb"; then
        echo "   → 使用TEB局部规划器"
    else
        echo -e "${YELLOW}⚠️${NC}  未识别的规划器类型"
    fi
else
    echo -e "${RED}❌${NC} 无法获取局部规划器配置"
fi
echo ""

# === 6. 检查RL节点配置 ===
echo "【6. RL局部规划器检查】"
if rosnode list | grep -q "/geo_nav_node"; then
    echo -e "${GREEN}✅${NC} geo_nav_node节点运行中（RL局部规划器）"

    # 检查RL节点订阅的话题
    echo "   检查RL节点输入..."
    if rostopic info /geo_nav_node/cmd_vel 2>/dev/null | grep -q "Subscribers:"; then
        echo -e "${GREEN}✅${NC} RL节点发布速度指令"
    else
        echo -e "${YELLOW}⚠️${NC}  RL节点可能未发布速度指令"
    fi
else
    echo -e "${RED}❌${NC} geo_nav_node节点未运行"
    echo -e "${YELLOW}→${NC} 当前使用传统DWA规划器（非RL方案）"
fi
echo ""

# === 7. 检查SLAM建图状态 ===
echo "【7. SLAM建图状态】"
if rosnode list | grep -q "/slam_gmapping"; then
    echo -e "${GREEN}✅${NC} slam_gmapping节点运行中"

    # 检查地图发布
    if rostopic list | grep -q "/map"; then
        echo -e "${GREEN}✅${NC} /map话题已发布"

        # 检查地图分辨率
        MAP_RESOLUTION=$(rosparam get /map_resolution 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "   地图分辨率: $MAP_RESOLUTION m/pixel"
        fi
    else
        echo -e "${RED}❌${NC} /map话题未发布"
    fi

    # 检查是否有地图更新
    echo "   检查地图更新..."
    MAP_COUNT=$(rostopic hz /map 2>/dev/null | head -3 | grep -oP 'average: \K[\d.]+' | head -1)
    if [ ! -z "$MAP_COUNT" ]; then
        echo -e "${GREEN}✅${NC} 地图更新中，频率约 ${MAP_COUNT} Hz"
    else
        echo -e "${YELLOW}⚠️${NC}  无法检测地图更新频率"
    fi
else
    echo -e "${RED}❌${NC} slam_gmapping节点未运行"
    echo -e "${YELLOW}→${NC} 无法进行SLAM建图"
fi
echo ""

# === 8. 检查全局路径规划器 ===
echo "【8. 全局路径规划器检查】"
GLOBAL_PLANNER=$(rosparam get /move_base/base_global_planner 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅${NC} 全局规划器类型: $GLOBAL_PLANNER"

    if echo "$GLOBAL_PLANNER" | grep -q "navfn"; then
        echo "   → 使用NavFn全局规划器"
    elif echo "$GLOBAL_PLANNER" | grep -q "dijkstra"; then
        echo "   → 使用Dijkstra全局规划器"
    else
        echo "   → 使用自定义全局规划器"
    fi
else
    echo -e "${RED}❌${NC} 无法获取全局规划器配置"
fi
echo ""

# === 9. 系统架构总结 ===
echo "【9. 系统架构总结】"
echo "当前运行架构："

if rosnode list | grep -q "/slam_gmapping"; then
    echo -e "  ${GREEN}SLAM建图${NC} → ✅ 启用"
else
    echo -e "  ${RED}SLAM建图${NC} → ❌ 未启用"
fi

if rosnode list | grep -q "/move_base"; then
    echo -e "  ${GREEN}全局规划${NC} → ✅ 启用 (move_base)"
else
    echo -e "  ${RED}全局规划${NC} → ❌ 未启用"
fi

if rosnode list | grep -q "/geo_nav_node"; then
    echo -e "  ${GREEN}局部规划${NC} → ✅ RL模型 (geo_nav_node)"
elif rosparam get /move_base/base_local_planner 2>/dev/null | grep -q "dwa"; then
    echo -e "  ${YELLOW}局部规划${NC} → ⚠️  DWA传统算法 (未使用RL)"
else
    echo -e "  ${RED}局部规划${NC} → ❌ 未启用"
fi
echo ""

# === 10. 使用建议 ===
echo "【10. 下一步操作建议】"
echo ""
if rosnode list | grep -q "/geo_nav_node"; then
    echo -e "${GREEN}当前使用RL局部规划器${NC}"
    echo "→ 发送导航目标测试："
    echo "  rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \\"
    echo "    '{header: {frame_id: \"map\"}, pose: {position: {x: 2.0, y: 0.0}, orientation: {w:1.0}}}' --once"
else
    echo -e "${YELLOW}当前未使用RL局部规划器${NC}"
    echo "→ 启动RL节点（在新终端）："
    echo "  conda activate env_isaaclab"
    echo "  python catkin_ws/src/dashgo_rl/scripts/geo_nav_node.py"
    echo ""
    echo "→ 或先测试DWA传统导航："
    echo "  rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \\"
    echo "    '{header: {frame_id: \"map\"}, pose: {position: {x: 2.0, y: 0.0}, orientation: {w:1.0}}}' --once"
fi
echo ""

echo "=== 诊断完成 ==="
