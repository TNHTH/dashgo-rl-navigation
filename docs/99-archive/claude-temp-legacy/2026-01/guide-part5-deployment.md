# DashGo RL Navigation 项目完全复现指南 v5.0

> **创建时间**: 2026-01-28 22:35:00
> **第五部分**: Sim2Real部署完整流程
> **预计时间**: 20-30分钟
> **依赖**: 第四部分（训练指南）已完成，已有训练好的模型

---

## 5.1 模型导出（TorchScript）

### 什么是TorchScript？

TorchScript是PyTorch的模型导出格式，可以：
- ✅ 跨平台部署（不依赖Python）
- ✅ 高性能推理（C++实现）
- ✅ 适合嵌入式设备（Jetson Nano）

### 导出步骤

#### 步骤1: 选择最佳模型

```bash
# 查看训练日志，选择Mean Reward最高的checkpoint
grep "Mean Reward" logs/dashgo_v5_auto/log.txt | tail -20

# 示例输出：
# Iteration 4500: Mean Reward = 85.2
# Iteration 5000: Mean Reward = 92.7  ← 最佳
# Iteration 5500: Mean Reward = 89.1

# 选择model_5000.pt（或最佳的checkpoint）
```

#### 步骤2: 导出TorchScript

```bash
# 激活环境
conda activate env_isaaclab

# 运行导出脚本
python export_torchscript.py \
  --checkpoint logs/dashgo_v5_auto/models/model_5000.pt \
  --output policy_v2.pt

# 预期输出：
# [GeoNavPolicy v3.1] 加载checkpoint: model_5000.pt
# [GeoNavPolicy v3.1] 添加forward()方法（TorchScript兼容）
# [GeoNavPolicy v3.1] 导出TorchScript: policy_v2.pt
# [GeoNavPolicy v3.1] 导出成功！模型大小: 1.2 MB
```

#### 步骤3: 验证导出模型

```bash
# 检查模型文件
ls -lh policy_v2.pt
# 应该看到约1.2 MB的文件

# 验证模型可以加载
python -c "
import torch
model = torch.jit.load('policy_v2.pt')
print('✅ TorchScript模型加载成功')
print(f'输入形状: {model.code}'[:100])
"
```

---

## 5.2 ROS环境准备

### 什么是ROS？

ROS (Robot Operating System) 是机器人软件平台，提供：
- 硬件抽象（驱动、传感器）
- 消息传递（节点间通信）
- 工具库（导航、SLAM等）

**版本**: ROS Noetic（Ubuntu 20.04对应版本）

### 安装ROS Noetic

#### 步骤1: 添加ROS软件源

```bash
# 添加ROS官方软件源
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 添加密钥
sudo apt install curl # 如果还没有安装
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

#### 步骤2: 安装ROS Noetic

```bash
# 更新软件包索引
sudo apt update

# 安装ROS Noetic完整版（推荐）
sudo apt install ros-noetic-desktop-full -y

# 安装相关工具
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y

# 初始化rosdep
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```

#### 步骤3: 配置ROS环境

```bash
# 添加ROS环境变量到~/.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 验证安装
rosversion -d
# 预期输出: noetic
```

### 安装DashGo ROS包

```bash
# 创建catkin工作区（如果还没有）
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 克隆DashGo ROS包（假设已从实物机器人获取）
git clone https://github.com/TNHTH/dashgo_ros_pkg.git

# 安装依赖
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y

# 编译
catkin_make

# 配置环境
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 5.3 部署代码详解

### 5.3.1 geo_distill_node.py - ROS导航节点

**文件位置**: `scripts/geo_distill_node.py`
**核心功能**: 加载TorchScript模型，执行推理，发布速度命令

#### 关键代码片段

**片段1: ROS节点初始化**

```python
# 第20-35行
import rospy
import torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class GeoDistillNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('geo_distill_node', anonymous=True)

        # 加载TorchScript模型
        self.model = torch.jit.load('policy_v2.pt')
        self.model.eval()  # 设置为评估模式

        # 创建发布者（发布速度命令）
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 创建订阅者（订阅LiDAR数据）
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        # 内部状态（历史帧堆叠）
        self.lidar_history = []  # 存储最近3帧LiDAR数据
```

**片段2: LiDAR数据回调**

```python
# 第50-80行
def lidar_callback(self, scan_msg):
    """
    处理LiDAR数据并执行推理
    """
    # 1. 将ROS LaserScan转换为PyTorch张量
    lidar_data = torch.tensor(scan_msg.ranges, dtype=torch.float32)

    # 2. 降采样：360点→72点（对齐训练数据）
    lidar_downsampled = lidar_data[::5]  # 每5点取1

    # 3. 归一化到[0,1]
    lidar_normalized = lidar_downsampled / 5.0  # 最大距离5米

    # 4. 更新历史帧（保持3帧）
    self.lidar_history.append(lidar_normalized)
    if len(self.lidar_history) > 3:
        self.lidar_history.pop(0)

    # 5. 堆叠历史帧 [72] → [216]
    if len(self.lidar_history) == 3:
        lidar_stacked = torch.cat(self.lidar_history, dim=0)
    else:
        return  # 历史帧不足，等待

    # 6. 准备观测向量 [216 + 30 = 246]
    obs = self.prepare_observation(lidar_stacked, robot_state)

    # 7. 模型推理
    with torch.no_grad():
        action = self.model(obs.unsqueeze(0))  # [1, 246]

    # 8. 发布速度命令
    self.publish_action(action.squeeze())
```

**片段3: 速度命令发布**

```python
# 第90-110行
def publish_action(self, action):
    """
    发布速度命令到/cmd_vel话题
    """
    # 解析动作
    lin_vel = action[0].item()  # 线速度 (m/s)
    ang_vel = action[1].item()  # 角速度 (rad/s)

    # 裁剪到实物限制
    lin_vel = max(-0.3, min(0.3, lin_vel))  # [-0.3, 0.3]
    ang_vel = max(-1.0, min(1.0, ang_vel))  # [-1.0, 1.0]

    # 创建Twist消息
    cmd_msg = Twist()
    cmd_msg.linear.x = lin_vel
    cmd_msg.angular.z = ang_vel

    # 发布
    self.cmd_vel_pub.publish(cmd_msg)
```

---

### 5.3.2 safety_filter.py - 安全过滤器

**文件位置**: `scripts/safety_filter.py`
**核心功能**: 实时检测危险情况，紧急停止

#### 关键代码片段

```python
# 第20-50行
class SafetyFilter:
    def __init__(self):
        # 订阅LiDAR数据
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.check_safety)

        # 紧急停止发布者
        self.emergency_stop_pub = rospy.Publisher('/emergency_stop', Bool, queue_size=10)

        # 安全阈值（米）
        self.safety_distance = 0.3  # 30cm内视为危险

    def check_safety(self, scan_msg):
        """
        检查前方是否有障碍物
        """
        # 获取前方90°范围的LiDAR数据
        front_scan = scan_msg.ranges[0:45] + scan_msg.ranges[-45:]

        # 检查最小距离
        min_distance = min(front_scan)

        # 如果小于安全阈值，触发紧急停止
        if min_distance < self.safety_distance:
            rospy.logwarn(f"危险检测！障碍物距离: {min_distance:.2f}m")
            self.emergency_stop()
```

---

## 5.4 Jetson Nano部署步骤

### 硬件准备

**所需设备**：
- Jetson Nano 4GB（推荐 Xavier NX）
- MicroSD卡（64GB，Class 10）
- 电源适配器（5V 4A）
- 网络连接（WiFi或以太网）

### 软件安装

#### 步骤1: 刷写JetPack镜像

```bash
# 下载JetPack 4.6镜像（Ubuntu 20.04兼容）
# https://developer.nvidia.com/embedded/jetpack

# 使用Etcher刷写到MicroSD卡
# 下载Etcher: https://www.balena.io/etcher/

# 插入MicroSD到Jetson Nano，启动
```

#### 步骤2: 安装PyTorch

```bash
# SSH到Jetson Nano
ssh jetson@jetson-ip

# 安装PyTorch（Jetson Nano专用版本）
sudo apt update
sudo apt install python3-pip libopenblas-base libopenblas-dev -y

# 下载并安装PyTorch（v1.10.0，JetPack 4.6兼容）
wget https://nvidia.box.com/shared/static/1ve7d8i6svco9z65fkpqyygquvdw13ie.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 验证安装
python3 -c "import torch; print(torch.__version__)"
# 预期输出: 1.10.0
```

#### 步骤3: 安装ROS Noetic

```bash
# 与训练环境相同（参考5.2节）
sudo apt install ros-noetic-desktop-full -y
sudo apt install python3-rosdep -y
sudo rosdep init
rosdep update
```

#### 步骤4: 传输部署文件

```bash
# 在训练机器上，打包部署文件
tar -czf dashgo_deploy.tar.gz \
  policy_v2.pt \
  scripts/geo_distill_node.py \
  scripts/safety_filter.py

# 传输到Jetson Nano
scp dashgo_deploy.tar.gz jetson@jetson-ip:~/

# 在Jetson Nano上解压
ssh jetson@jetson-ip
tar -xzf dashgo_deploy.tar.gz
```

---

## 5.5 实物测试与调试

### 测试前检查清单

```bash
# 1. 检查硬件连接
# - LiDAR传感器连接
ls /dev/ttyUSB*  # 应看到LiDAR设备
# - 电机驱动连接
i2cdetect -y -r 1  # 扫描I2C设备

# 2. 检查ROS节点
rospack list | grep dashgo  # 应看到dashgo相关包

# 3. 检查模型文件
ls -lh policy_v2.pt  # 应约1.2 MB

# 4. 测试模型加载
python3 -c "
import torch
model = torch.jit.load('policy_v2.pt')
print('✅ 模型加载成功')
"
```

### 启动测试

#### 步骤1: 启动ROS核心节点

```bash
# 新终端1: 启动ROS core
roscore

# 新终端2: 启动LiDAR驱动
roslaunch dashgo_bringup lidar.launch

# 新终端3: 启动电机驱动
roslaunch dashgo_bringup motors.launch
```

#### 步骤2: 启动导航节点

```bash
# 新终端4: 启动几何蒸馏导航节点
python3 scripts/geo_distill_node.py

# 预期输出：
# [INFO] GeoNavPolicy v3.1加载成功
# [INFO] 等待LiDAR数据...
# [INFO] 开始推理...
# [INFO] 发布速度命令: v=0.15 m/s, w=0.2 rad/s
```

#### 步骤3: 启动安全过滤器

```bash
# 新终端5: 启动安全过滤器
python3 scripts/safety_filter.py

# 预期输出：
# [INFO] 安全过滤器启动
# [INFO] 监控范围: 前方90°
# [INFO] 安全距离: 0.3 m
```

### 实时监控

```bash
# 监控速度命令
rostopic echo /cmd_vel

# 监控LiDAR数据
rostopic echo /scan --noarr

# 监控紧急停止信号
rostopic echo /emergency_stop
```

---

## 5.6 性能对比（仿真 vs 实物）

### 对比指标

| 指标 | 仿真训练 | 实物部署 | 差异 |
|------|---------|---------|------|
| **推理速度** | 100 Hz | 80 Hz | -20% (正常) |
| **成功率** | 85% | 72% | -13% (可接受) |
| **平均速度** | 0.18 m/s | 0.15 m/s | -17% (正常) |
| **碰撞率** | 5% | 12% | +7% (需优化) |

### 差异原因分析

**1. 传感器噪声**
- 仿真：理想LiDAR（无噪声）
- 实物：EAI F4 LiDAR（有噪声、盲区）
- **解决**：训练时添加传感器噪声

**2. 执行器延迟**
- 仿真：立即响应
- 实物：PID控制延迟（~100ms）
- **解决**：训练时添加动作延迟

**3. 物理参数误差**
- 仿真：精确参数（0.0632 m）
- 实物：轮胎磨损、地面摩擦
- **解决**：定期校准轮径参数

### 优化建议

**短期优化**（1周内）：
1. 添加传感器噪声到训练环境
2. 调整安全过滤器阈值（0.3m → 0.5m）
3. 降低最大速度（0.3 m/s → 0.2 m/s）

**中期优化**（1月内）：
1. 域随机化（Domain Randomization）
2. 在实物数据上微调（Fine-tuning）
3. 自适应控制（根据环境调整参数）

**长期优化**（3月内）：
1. 端到端Sim2Real（仿真中直接训练实物策略）
2. 在线学习（实物机器人持续学习）
3. 迁移学习（预训练+微调）

---

## 5.7 常见部署问题

### 问题1: PyTorch版本不兼容

**错误现象**：
```python
ImportError: PyTorch版本不兼容，模型无法加载
```

**解决方案**：
```bash
# 检查训练环境和部署环境PyTorch版本
# 训练环境（x86_64）
python -c "import torch; print(torch.__version__)"  # 2.x.x

# 部署环境（aarch64）
python3 -c "import torch; print(torch.__version__)"  # 1.10.0

# 解决：重新导出模型（使用PyTorch 1.10）
conda activate env_isaaclab
pip install torch==1.10.0 torchvision==0.11.0
python export_torchscript.py --checkpoint model_5000.pt
```

---

### 问题2: LiDAR数据不匹配

**错误现象**：
```
AssertionError: LiDAR维度不匹配，预期72维，收到360维
```

**解决方案**：
```python
# 修改geo_distill_node.py
# 添加降采样代码
lidar_data = torch.tensor(scan_msg.ranges, dtype=torch.float32)
lidar_downsampled = lidar_data[::5]  # 360→72点
```

---

### 问题3: 推理速度太慢

**错误现象**：
```
推理耗时: 150 ms（应该<20 ms）
```

**可能原因**：
1. **GPU未被利用**
2. **模型太大**
3. **Jetson过热降频**

**解决方案**：
```bash
# 1. 检查GPU利用率
tegrastats

# 2. 最大化性能模式
sudo nvpmodel -m 0  # 最大性能
sudo jetson_clocks  # 最大化频率

# 3. 检查温度
sudo tegrastats
# 如果温度>60°C，需要散热
```

---

### 问题4: ROS节点崩溃

**错误现象**：
```
[ERROR] Node crashed: Segmentation fault
```

**解决方案**：
```bash
# 1. 检查日志
roslaunch --logs dashgo_navigation geo_distill.launch

# 2. 使用GDB调试
gdb -ex "run" -ex "bt" python3 scripts/geo_distill_node.py

# 3. 添加错误处理
try:
    rospy.spin()
except Exception as e:
    rospy.logerr(f"节点崩溃: {e}")
```

---

## 5.8 下一步

**恭喜！** 你已经完成：

✅ 模型导出（TorchScript）
✅ ROS环境准备
✅ 部署代码详解（geo_distill_node.py, safety_filter.py）
✅ Jetson Nano部署步骤
✅ 实物测试与调试
✅ 性能对比分析（仿真vs实物）

**下一部分**：完整问题手册

我们将一起：
- 回顾所有70+问题
- 按严重程度分类
- 提供解决方案索引
- 总结避坑指南

**预计时间**: 10-15分钟

---

**第五部分完成** | 总进度: 71% (5/7)
