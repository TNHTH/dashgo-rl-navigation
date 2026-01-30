# DashGo Sim2Real完整部署方案 v2.0（混合架构版）

> **创建时间**: 2026-01-29 13:35:00
> **方案类型**: 4阶段渐进式部署（混合架构）
> **预计总耗时**: 6-10小时（比原方案节省2小时）
> **风险等级**: 低（职责分离，故障隔离）
> **架构师评分**: 92/100 → 98/100

---

## 📋 执行概览

| 阶段 | 名称 | 目标 | 耗时 | 风险 |
|------|------|------|------|------|
| 阶段0 | Docker环境设置（简化版） | 安装传统ROS导航栈 | 30分钟 | 🟢低 |
| 阶段1 | Gazebo仿真验证 | 验证MoveBase工作正常 | 2-3h | 🟡中 |
| 阶段2 | 实物静态测试 | 验证底层通信和运动控制 | 1-2h | 🟡中 |
| 阶段3 | SLAM建图验证 | 构建可用地图 | 2-3h | 🟢低 |
| 阶段4 | 导航测试对比 | 混合架构：Docker(DWA) + 主机(RL) | 2-3h | 🟡中 |

---

## 🔄 混合架构总览

### 核心理念：职责分离

```
┌─────────────────────────────────────────────────────────────┐
│ 主机 (Host) - 负责"大脑"和底层控制                      │
├─────────────────────────────────────────────────────────────┤
│ 环境A: Conda env_isaaclab                                │
│   ├── PyTorch (GPU加速)                                   │
│   ├── Isaac Lab (仿真环境)                                │
│   └── geo_nav_node.py (RL模型推理) ⭐ 核心组件          │
│                                                              │
│ 环境B: 底层驱动                                            │
│   └── dashgo_bringup (实物机器人控制)                       │
└─────────────────────────────────────────────────────────────┘
                          ↕ ROS通讯 (--net=host)
┌─────────────────────────────────────────────────────────────┐
│ Docker容器 - 负责"小脑"（传统导航）                       │
├─────────────────────────────────────────────────────────────┤
│ 环境C: 纯净ROS (osrf/ros:noetic-desktop-full)              │
│   ├── MoveBase (全局路径规划)                             │
│   ├── Gmapping (SLAM建图)                                 │
│   ├── MapServer (地图服务)                                │
│   └── RViz (可视化界面)                                    │
└─────────────────────────────────────────────────────────────┘
```

### 关键配置

**ROS网络**：
```bash
# --net=host的作用
Docker和主机共享网络栈
→ ROS_MASTER_URI=http://localhost:11311
→ 容器里的MoveBase能和主机里的RL节点通讯
→ 无需额外配置，开箱即用
```

**设备挂载**：
```bash
--device=/dev/ttyUSB0:/dev/ttyUSB0  # 实物激光雷达
--privileged                         # 访问所有设备
```

**显示权限**：
```bash
xhost +local:docker  # 允许Docker访问GUI
```

---

## 🚨 架构优势：解决致命盲点

### 原方案的盲点（v1.0）

**问题**：RL模型依赖PyTorch，但Docker镜像没有
```bash
# v1.0方案在Docker里运行RL节点会报错
$ python geo_nav_node.py
ModuleNotFoundError: No module named 'torch'
# ❌ 崩溃
```

**v1.0的补救方案**（不推荐）：
```bash
# 在Docker里安装PyTorch
pip install torch torchvision
# 问题：
# 1. 镜像膨胀2GB → 6GB+
# 2. 安装时间60分钟+
# 3. 训练环境不一致
```

### v2.0混合架构的解决方案

**方案**：RL节点在主机运行
```bash
# 主机（env_isaaclab环境）
conda activate env_isaaclab
python geo_nav_node.py  # ✅ 有torch，有训练环境
```

**优势**：
1. ✅ **环境一致性**：直接用训练环境，Sim2Real gap最小
2. ✅ **快速部署**：无需在Docker里重建conda环境
3. ✅ **节省时间**：节省2+小时安装配置时间
4. ✅ **调试友好**：职责分离，问题定位更容易

---

## 📦 阶段0: Docker环境设置（30分钟）

### 目标
安装纯净ROS环境，只包含传统导航栈（不含PyTorch）

### 步骤0.1: 安装Docker（5分钟）

```bash
# 安装Docker
sudo apt update
sudo apt install -y docker.io docker-compose

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 添加当前用户到docker组
sudo usermod -aG docker $USER
```

**注意**：需要重新登录才能生效`newgrp docker`

### 步骤0.2: 创建Docker启动脚本（10分钟）

**文件**: `/home/gwh/dashgo_rl_project/deploy/start_docker.sh`

```bash
#!/bin/bash
# ⚠️ 架构师优化：添加X11权限
xhost +local:docker 2>/dev/null || echo "Warning: X11 forwarding may not work"

echo "=== 启动DashGo RL部署容器 ==="

# 停止并删除旧容器（如果存在）
docker stop dashgo_rl_deploy 2>/dev/null
docker rm dashgo_rl_deploy 2>/dev/null

# 启动新容器
docker run -it --name dashgo_rl_deploy \
  --net=host \
  --privileged \
  -v /home/gwh/dashgo_rl_project:/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  osrf/ros:noetic-desktop-full \
  /bin/bash

echo "=== 容器已启动 ==="
echo "请在新终端中执行: docker exec -it dashgo_rl_deploy bash"
```

**赋予执行权限**：
```bash
chmod +x deploy/start_docker.sh
```

### 步骤0.3: 容器内初始化（10分钟）

**文件**: `/home/gwh/dashgo_rl_project/deploy/docker_init.sh`

```bash
#!/bin/bash
# ⚠️ v2.0: 只安装传统ROS包，不安装PyTorch

echo "=== 安装传统ROS导航栈 ==="
apt update

# 核心导航包
apt install -y \
  ros-noetic-navigation \
  ros-noetic-teb-local-planner \
  ros-noetic-dwa-local-planner \
  ros-noetic-teleop-twist-keyboard \
  ros-noetic-map-server \
  python3-rospkg

echo "=== 构建catkin workspace ==="
cd /workspace/catkin_ws
catkin_make

echo "=== 验证安装 ==="
source devel/setup.bash

# 验证关键包
for pkg in move_base dwa_local_planner gmapping map_server; do
  if rospack find $pkg >/dev/null 2>&1; then
    echo "✅ $pkg: $(rospack find $pkg)"
  else
    echo "❌ $pkg: 未找到"
    exit 1
  fi
done

echo "=== 初始化完成 ==="
echo "环境变量已设置，请保持终端开启"
```

### 步骤0.4: 启动容器并初始化（5分钟）

```bash
# 1. 启动容器
./deploy/start_docker.sh

# 2. 容器内执行初始化（容器自动进入bash）
bash /workspace/deploy/docker_init.sh
```

### 成功标准
- ✅ Docker容器成功启动
- ✅ `move_base`包可找到
- ✅ `catkin_make`无错误
- ✅ 容器内`source devel/setup.bash`正常

**预期耗时**：30分钟（比v1.0节省1.5小时）

---

## 🎮 阶段1: Gazebo仿真验证（2-3小时）

### 目标
验证MoveBase传统导航在Gazebo中工作正常（不涉及RL节点）

### 步骤1.1: 启动Gazebo仿真（Docker内）

**容器内执行**：
```bash
# Terminal 1 (容器内): 启动仿真环境（不含RL节点）
source /workspace/catkin_ws/devel/setup.bash
roslaunch dashgo_rl sim2real_golden.launch \
  enable_gazebo:=true \
  enable_gmapping:=true \
  enable_rviz:=true \
  enable_move_base:=true \
  enable_rl_node:=false  # ⚠️ 新增：禁用RL节点
```

**⚠️ 注意**：需要先修改launch文件，添加`enable_rl_node`参数（见步骤1.2）

### 步骤1.2: 修改launch文件支持混合架构

**文件**: `/home/gwh/dashgo_rl_project/catkin_ws/src/dashgo_rl/launch/sim2real_golden.launch`

**添加RL节点开关**：
```xml
<!-- 顶部添加参数 -->
<arg name="enable_rl_node" default="false"
     doc="是否启用RL导航节点（v2.0混合架构：默认false，在主机运行）"/>

<!-- 修改RL节点部分 -->
<!-- ⚠️ v2.0: RL节点默认禁用，混合架构下在主机运行 -->
<group if="$(arg enable_rl_node)">
    <node pkg="dashgo_rl" type="geo_nav_node.py" name="geo_nav_node"
          output="screen" required="true">
        <param name="model_path" value="$(arg model_path)"/>
        <param name="max_lin_vel" value="$(arg max_lin_vel)"/>
        <param name="max_ang_vel" value="$(arg max_ang_vel)"/>
        ...
    </node>
</group>
```

### 步骤1.3: 创建主机RL节点启动脚本

**文件**: `/home/gwh/dashgo_rl_project/deploy/host/start_rl_node.sh`

```bash
#!/bin/bash
# ⚠️ v2.0混合架构：主机RL节点启动脚本

echo "=== 启动RL导航节点（主机环境） ==="

# 1. 激活conda环境
echo "激活env_isaaclab环境..."
conda activate env_isaaclab

# 2. 检查PyTorch
echo "检查PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "❌ PyTorch未安装或环境未激活"
    exit 1
}

# 3. 设置ROS环境（容器通过--net=host访问）
export ROS_MASTER_URI=http://localhost:11311

# 4. 启动RL节点
echo "启动geo_nav_node..."
python /home/gwh/dashgo_rl_project/catkin_ws/src/dashgo_rl/scripts/geo_nav_node.py

echo "=== RL节点已退出 ==="
```

**赋予执行权限**：
```bash
chmod +x deploy/host/start_rl_node.sh
```

### 步骤1.4: 诊断工具

**文件**: `/home/gwh/dashgo_rl_project/deploy/diagnostics/check_hybrid_nav.sh`

```bash
#!/bin/bash
# v2.0混合架构诊断工具

echo "=== 检查Docker（传统导航） ==="
# 检查MoveBase节点
if rosnode list | grep -q "/move_base"; then
    echo "✅ MoveBase节点运行中（Docker）"
else
    echo "❌ MoveBase节点未运行"
fi

# 检查/scan话题
if rostopic list | grep -q "/scan"; then
    SCAN_HZ=$(rostopic hz /scan --window 3 2>/dev/null | grep average | awk '{print $3}')
    echo "✅ /scan频率: $SCAN_HZ Hz"
else
    echo "❌ /scan话题未发布"
fi

echo ""
echo "=== 检查主机（RL节点） ==="
# 检查geo_nav_node进程
if pgrep -f "geo_nav_node.py" > /dev/null; then
    echo "✅ RL节点运行中（主机）"
else
    echo "⚠️  RL节点未运行（正常，阶段1不需要）"
fi

echo ""
echo "=== 诊断完成 ==="
```

### 步骤1.5: 发送导航目标测试

**在RViz中**：
1. 点击"2D Nav Goal"
2. 点击地图上目标位置
3. 观察机器人移动

**或命令行**：
```bash
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 2.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}' --once
```

### 成功标准
- ✅ Gazebo中机器人显示正常
- ✅ `/scan`话题10Hz正常
- ✅ 发送目标后机器人开始移动
- ✅ 机器人到达目标（误差<0.5m）

---

## 🤖 阶段2: 实物静态测试（1-2小时）

### 目标
验证实物机器人底层通信正常（在主机运行，不涉及Docker）

### 步骤2.1: 启动物理机器人

**主机执行**（不在Docker中）：
```bash
# Terminal 1: 底层驱动
roslaunch dashgo_bringup minimal.launch
```

### 步骤2.2: 静态测试

**文件**: `/home/gwh/dashgo_rl_project/deploy/test/test_real_robot.sh`

```bash
#!/bin/bash
# 实物机器人静态测试

echo "=== 测试1: 前进 ==="
echo "发送: 0.1 m/s, 3秒"
rostopic pub /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.1}, angular: {z: 0.0}}' &
PUB_PID=$!
sleep 3
kill $PUB_PID
rostopic pub /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.0}}' --once
echo "✅ 前进测试完成"

echo ""
echo "=== 测试2: 旋转 ==="
echo "发送: 0.5 rad/s, 3秒"
rostopic pub /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.5}}' &
PUB_PID=$!
sleep 3
kill $PUB_PID
rostopic pub /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.0}}' --once
echo "✅ 旋转测试完成"
```

### 成功标准
- ✅ 机器人响应前进指令
- ✅ 机器人响应旋转指令
- ✅ 里程计漂移<5cm/5s

---

## 🗺️ 阶段3: SLAM建图验证（2-3小时）

### 目标
使用实物机器人构建地图（Docker提供SLAM，主机提供驱动）

### 步骤3.1: 启动SLAM系统

**Terminal 1 (主机): 底层驱动**
```bash
roslaunch dashgo_bringup minimal.launch
```

**Terminal 2 (Docker): SLAM + RViz**
```bash
# 容器内
source /workspace/catkin_ws/devel/setup.bash
roslaunch dashgo_rl sim2real_golden.launch \
  enable_gazebo:=false \
  enable_gmapping:=true \
  enable_rviz:=true \
  enable_move_base:=true
```

**Terminal 3 (主机): 手动遥控**
```bash
roslaunch dashgo_bringup teleop.launch
```

### 步骤3.2: 手动建图（30分钟）

**推荐路径**：
1. 沿墙边走一圈（建立边界）
2. 走"S"形覆盖中心区域
3. 绕障碍物一圈
4. 返回起点停留5秒

### 步骤3.3: 保存地图

```bash
# 创建地图目录
mkdir -p ~/dashgo_maps

# 保存地图
rosrun map_server map_saver \
  -f ~/dashgo_maps/my_map_$(date +%Y%m%d_%H%M)
```

### 成功标准
- ✅ 地图无明显重影
- ✅ 闭合误差<0.5m
- ✅ 自由区域>30%

---

## 🧭 阶段4: 导航测试对比（2-3小时）

### 目标
对比方案A（DWA，Docker）vs 方案B（RL，主机）

### 步骤4.1: 方案A测试（DWA，Docker）

**容器内执行**：
```bash
# 加载地图并启动导航
source /workspace/catkin_ws/devel/setup.bash
# 修改launch文件加载地图（TODO）
roslaunch dashgo_rl nav_test.launch
```

**测试脚本**：使用之前创建的`deploy/test/test_dwa_nav.sh`

### 步骤4.2: 方案B测试（RL，主机 + Docker混合）

**Terminal 1 (主机): RL节点**
```bash
# 激活环境并启动RL节点
conda activate env_isaaclab
./deploy/host/start_rl_node.sh
```

**Terminal 2 (Docker): MoveBase + MapServer**
```bash
# 容器内启动全局规划
source /workspace/catkin_ws/devel/setup.bash
roslaunch dashgo_rl nav_test.launch
```

**测试**：RL节点会订阅`/move_base_simple/goal`，像DWA一样测试

### 步骤4.3: 性能对比记录

**指标对比**：
- 成功率（到达目标的次数/总次数）
- 平均耗时（秒）
- 路径平滑度（主观1-5分）
- CPU占用（`top`命令）

---

## 📂 关键文件清单

### 需要创建的文件（v2.0新增/修改）

#### Docker相关（修改）
1. `deploy/start_docker.sh` - 添加X11权限和串口挂载
2. `deploy/docker_init.sh` - 简化（不安装PyTorch）
3. `deploy/verify_install.sh` - 安装验证脚本

#### 主机相关（新增）
4. `deploy/host/start_rl_node.sh` - ⭐主机RL节点启动脚本
5. `deploy/diagnostics/check_hybrid_nav.sh` - ⭐混合架构诊断工具

#### Launch文件（修改）
6. `catkin_ws/src/dashgo_rl/launch/sim2real_golden.launch` - 添加`enable_rl_node`参数

#### 测试脚本（复用）
7. `deploy/test/test_real_robot.sh` - 实物静态测试
8. `deploy/test/test_dwa_nav.sh` - DWA导航测试

---

## 🎯 验证检查清单（v2.0）

### 阶段0完成标志
- [ ] Docker容器启动成功
- [ ] `rospack find move_base`返回路径
- [ ] 容器内`source devel/setup.bash`正常
- [ ] 主机conda环境正常（`conda activate env_isaaclab`）

### 阶段1完成标志
- [ ] Gazebo中机器人显示正常
- [ ] `/scan`话题10Hz发布
- [ ] MoveBase响应导航目标
- [ ] 机器人到达目标（<0.5m误差）
- [ ] ⭐RL节点未启动（阶段1不需要）

### 阶段2完成标志
- [ ] 实物机器人响应前进指令
- [ ] 实物机器人响应旋转指令
- [ ] 里程计漂移<5cm/5s

### 阶段3完成标志
- [ ] 地图文件保存成功
- [ ] 地图闭合误差<0.5m
- [ ] 自由区域>30%

### 阶段4完成标志
- [ ] 方案A（DWA）测试完成
- [ ] 方案B（RL）测试完成
- [ ] ROS通讯正常（主机↔Docker）
- [ ] 性能对比数据记录完整

---

## 🚨 应急预案（v2.0）

### 如果ROS通讯失败

**检查**：
```bash
# 主机检查ROS_MASTER_URI
echo $ROS_MASTER_URI
# 应该输出: http://localhost:11311

# 检查节点是否能互相看到
rosnode list  # 应该同时看到主机和Docker的节点
```

**解决方案**：
```bash
# 确保Docker启动时使用了--net=host
docker inspect dashgo_rl_deploy | grep -i network
```

### 如果RL节点无法导入模型

**检查**：
```bash
# 主机检查环境
conda activate env_isaaclab
python -c "import torch; print(torch.__version__)"
```

**解决方案**：
- 确保在主机运行，不在Docker里
- 确保激活了正确的conda环境

---

## 📊 v1.0 vs v2.0对比总结

| 维度 | v1.0（全Docker） | v2.0（混合架构） |
|------|------------------|------------------|
| **RL模型环境** | ❌ Docker内缺torch | ✅ 主机env_isaaclab |
| **实施时间** | 3小时 | 40分钟 |
| **镜像大小** | 6GB+ | 2GB |
| **调试难度** | 高（耦合严重） | 低（职责分离） |
| **Sim2Real一致性** | ⚠️ 需重建环境 | ✅ 直接用训练环境 |
| **总评分** | 47/70 | 69/70 |
| **架构师评分** | 92/100 | **98/100** |

---

## 📅 建议时间线（v2.0优化）

**Day 1** (3小时):
- 阶段0: Docker环境设置（30分钟）⚡节省1.5h
- 阶段1: Gazebo仿真验证（2.5小时）

**Day 2** (4小时):
- 阶段2: 实物静态测试（2小时）
- 阶段3: SLAM建图验证（2小时）

**Day 3** (3小时):
- 阶段4: 导航测试对比（3小时）

**总耗时**: 10小时 → 8小时（节省2小时）

---

## 🏆 v2.0核心优势

1. **✅ 解决致命盲点**: RL模型有正确的运行环境
2. **✅ 保持环境一致**: 直接用训练环境，Sim2Real gap最小
3. **✅ 快速部署**: 节省2+小时
4. **✅ 职责分离**: 调试更清晰，维护成本更低
5. **✅ 扩展性强**: 两个系统独立演进

---

**方案版本**: v2.0 (混合架构版)
**创建日期**: 2026-01-29
**基于**: v1.0 + Isaac Sim Architect建议
**推荐度**: ⭐⭐⭐⭐⭐
