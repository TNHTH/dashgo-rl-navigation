#!/bin/bash

# ============================================================================
# DashGo 机器人连接脚本 (增强版 v2.0)
# 包含：IP自动发现、时间同步检查、环境配置
# 架构师修正：添加时间同步检查逻辑
# ============================================================================

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 1. 智能获取本机IP（优先192.168网段）
get_local_ip() {
    MY_IP=$(ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d/ -f1 | grep '192.168' | head -n 1)

    # Fallback: 如果没有192.168，尝试其他网段
    if [ -z "$MY_IP" ]; then
        MY_IP=$(ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d/ -f1 | grep -E '^(172\.(1[6-9]|2[0-9]|3[01])\.|10\.)' | head -n 1)
    fi

    # 如果还是找不到，让用户手动输入
    if [ -z "$MY_IP" ]; then
        echo -e "${YELLOW}⚠️  无法自动识别IP，请输入本机IP:${NC}"
        echo "您的网卡IP列表："
        ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print "  - " $2}' | cut -d/ -f1
        echo ""
        echo -n "请输入本机IP: "
        read MY_IP
    fi
}

get_local_ip

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DashGo 远程连接配置 (v2.0)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "笔记本IP: ${BLUE}$MY_IP${NC}"
echo ""

# 2. 交互式输入机器人IP
if [ -n "$1" ]; then
    ROBOT_IP=$1
else
    echo -n "请输入机器人IP [默认 192.168.5.100]: "
    read ROBOT_IP
    ROBOT_IP=${ROBOT_IP:-192.168.5.100}
fi

echo -e "机器人IP: ${BLUE}$ROBOT_IP${NC}"
echo ""

# 3. 连通性检查
echo -n "📡 正在 Ping 机器人... "
if ping -c 1 -W 2 $ROBOT_IP &> /dev/null; then
    echo -e "${GREEN}✅ 成功${NC}"
else
    echo -e "${RED}❌ 失败！请检查网络连接。${NC}"
    exit 1
fi

# ============================================================================
# 4. [架构师修正] 时间同步检查 (Time Sync Check)
# ============================================================================
echo ""
echo -n "⏳ 正在检查与机器人的时间偏差... "

# 检查是否安装ntpdate
if ! command -v ntpdate &> /dev/null; then
    echo -e "${YELLOW}⚠️  未安装ntpdate，尝试安装...${NC}"
    sudo apt install ntpdate -y
fi

# 尝试使用 ntpdate 强制同步 (需要机器人端开启 ntp 服务)
if sudo ntpdate -u $ROBOT_IP &> /dev/null; then
    echo -e "${GREEN}✅ 时间同步成功${NC}"
else
    # 如果自动同步失败，检查时间偏差
    LOCAL_TS=$(date +%s)

    # 尝试通过 SSH 获取机器人时间 (需要配置免密或手动输入密码)
    # 注意：如果机器人没有SSH访问，此步可能失败
    ROBOT_TS_STR=$(ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no $ROBOT_IP date +%s 2>/dev/null)

    if [ -z "$ROBOT_TS_STR" ]; then
        echo -e "${YELLOW}⚠️  无法通过 SSH 获取机器人时间${NC}"
        echo "请手动确认笔记本和机器人时间一致（误差<1秒）"
        echo ""
        echo "  笔记本时间: $(date)"
        echo "  机器人时间  : (请在机器人上运行 'date' 命令查看)"
        echo ""
        read -p "按回车键继续（如果您已确认时间已同步）..."
    else
        DIFF=$((LOCAL_TS - ROBOT_TS_STR))
        # 取绝对值
        ABS_DIFF=${DIFF#-}

        if [ $ABS_DIFF -gt 1 ]; then
            echo -e "${RED}❌ 严重警告！时间偏差为 $ABS_DIFF 秒${NC}"
            echo "TF 变换将失效。请务必同步时间："
            echo "  方法1: sudo date -s \"2026-01-28 10:00:00\""
            echo "  方法2: sudo ntpdate $ROBOT_IP"
            echo ""
            read -p "按回车键继续（如果您已手动修复）..."
        else
            echo -e "${GREEN}✅ 时间同步正常 (偏差 < 1s)${NC}"
        fi
    fi
fi
# ============================================================================

# 5. 导出 ROS 环境变量
export ROS_MASTER_URI=http://$ROBOT_IP:11311
export ROS_IP=$MY_IP

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ROS 环境已配置${NC}"
echo -e "${GREEN}========================================${NC}"
echo "   Master: $ROS_MASTER_URI"
echo "   IP    : $ROS_IP"
echo ""
echo -e "${GREEN}测试命令：${NC}"
echo "  rostopic list"
echo ""
echo -e "${GREEN}启动RL节点：${NC}"
echo "  1. 确保激活环境：conda activate dashgo_deploy"
echo "  2. 启动节点：roslaunch dashgo_rl real_control.launch"
echo ""
