#!/bin/bash
# Sim2Real快速部署脚本 - 架构师修正版
#
# 使用方法:
#   ./quickstart_deploy.sh [步骤]
#
# 步骤:
#   export   - 导出模型
#   build    - 编译ROS工作空间
#   test     - Gazebo测试
#   deploy   - 部署到Jetson

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 步骤1: 导出模型
export_model() {
    print_info "开始导出模型..."

    if [ ! -f "export_torchscript.py" ]; then
        print_error "export_torchscript.py不存在"
        exit 1
    fi

    # 检查Isaac Lab
    if [ ! -d "$HOME/IsaacLab" ]; then
        print_error "Isaac Lab未找到，请先安装"
        exit 1
    fi

    # 运行导出脚本
    $HOME/IsaacLab/isaaclab.sh -p export_torchscript.py

    # 验证导出
    if [ -f "catkin_ws/src/dashgo_rl/models/policy_torchscript.pt" ]; then
        size=$(du -h catkin_ws/src/dashgo_rl/models/policy_torchscript.pt | cut -f1)
        print_info "✅ 模型导出成功！大小: $size"
    else
        print_error "模型导出失败"
        exit 1
    fi
}

# 步骤2: 编译ROS工作空间
build_workspace() {
    print_info "开始编译ROS工作空间..."

    if [ ! -d "catkin_ws/src" ]; then
        print_error "ROS工作空间不存在"
        exit 1
    fi

    cd catkin_ws

    # 检查依赖
    print_info "检查ROS依赖..."
    if command -v rosdep &> /dev/null; then
        rosdep install --from-paths src --ignore-src -y || \
            print_warn "rosdep安装失败，继续编译..."
    fi

    # 编译
    print_info "编译中..."
    catkin_make

    # 加载环境
    source devel/setup.bash

    print_info "✅ 编译完成！"
    cd ..
}

# 步骤3: Gazebo测试
test_gazebo() {
    print_info "准备Gazebo测试..."

    # 检查节点
    if [ ! -f "catkin_ws/src/dashgo_rl/scripts/geo_nav_node.py" ]; then
        print_error "ROS节点不存在"
        exit 1
    fi

    print_info "请手动执行以下步骤："
    echo ""
    echo "1. Terminal 1 - 启动Gazebo:"
    echo "   cd ~/dashgo_rl_project/dashgo/1/1/nav"
    echo "   roslaunch launch/nav01.launch"
    echo ""
    echo "2. Terminal 2 - 启动RL Agent:"
    echo "   source ~/dashgo_rl_project/catkin_ws/devel/setup.bash"
    echo "   roslaunch dashgo_rl geo_nav.launch"
    echo ""
    echo "3. Terminal 3 - 发送目标点:"
    echo "   rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \\"
    echo "     \"header: frame_id: 'map'"
    echo "     pose: position: x: 2.0 y: 1.0 orientation: w: 1.0\""
    echo ""
}

# 步骤4: 部署到Jetson
deploy_jetson() {
    print_info "准备部署到Jetson..."

    JETSON_IP=${JETSON_IP:-"jetson@dashgo"}

    print_info "目标: $JETSON_IP"

    # 打包
    print_info "打包工作空间..."
    tar -czf catkin_ws.tar.gz catkin_ws/

    # 传输
    print_info "传输到Jetson..."
    scp catkin_ws.tar.gz $JETSON_IP:~/

    print_info "✅ 文件已传输！"
    print_info "请在Jetson上执行："
    echo ""
    echo "  cd ~"
    echo "  tar -xzf catkin_ws.tar.gz"
    echo "  cd catkin_ws"
    echo "  catkin_make"
    echo "  source devel/setup.bash"
    echo "  roslaunch dashgo_rl geo_nav.launch"
}

# 主函数
main() {
    case "$1" in
        export)
            export_model
            ;;
        build)
            build_workspace
            ;;
        test)
            test_gazebo
            ;;
        deploy)
            deploy_jetson
            ;;
        all)
            print_info "执行完整流程..."
            export_model
            build_workspace
            test_gazebo
            ;;
        *)
            echo "Sim2Real快速部署脚本 - 架构师修正版"
            echo ""
            echo "使用方法:"
            echo "  $0 export  - 导出模型"
            echo "  $0 build   - 编译ROS工作空间"
            echo "  $0 test    - Gazebo测试说明"
            echo "  $0 deploy  - 部署到Jetson"
            echo "  $0 all     - 执行完整流程"
            echo ""
            echo "环境变量:"
            echo "  JETSON_IP - Jetson IP地址 (默认: jetson@dashgo)"
            echo ""
            exit 1
            ;;
    esac
}

main "$@"
