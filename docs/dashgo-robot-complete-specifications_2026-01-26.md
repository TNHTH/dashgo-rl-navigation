# DashGo D1 机器人完整参数规格

> **更新时间**: 2026-01-26
> **机器人型号**: DashGo D1
> **制造商**: EAI (eaibot.cn)
> **开发基准**: Isaac Sim 4.5 + Ubuntu 20.04
> **参数来源**: ROS配置文件 + URDF模型 + 官方规格

---

## 📑 目录

- [物理尺寸参数](#物理尺寸参数)
- [执行器参数](#执行器参数)
- [运动参数](#运动参数)
- [传感器参数](#传感器参数)
- [通信参数](#通信参数)
- [电源系统](#电源系统)
- [导航参数](#导航参数)
- [差速驱动运动学](#差速驱动运动学)
- [配置使用方法](#配置使用方法)

---

## 📐 物理尺寸参数

### 机器人外形

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **主体直径** | 0.406 | m | URDF模型 |
| **主体高度** | 0.21 | m | URDF模型 |
| **主体质量** | 13.7 | kg | URDF模型 |
| **机器人半径** | 0.2 | m | ROS导航配置 |

### 轮子参数

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **轮子直径** | 0.1264 | m | ROS配置 |
| **轮子半径** | 0.0632 | m | 计算 |
| **轮子宽度** | 0.04 | m | URDF模型 |
| **单轮质量** | 1.5 | kg | URDF模型 |

### 轮子位置

| 参数 | 值 | 单位 | 说明 |
|------|-----|------|------|
| **轮距** | 0.3420 | m | 左右轮中心距 |
| **轮子X偏移** | 0.0 | m | 前后位置 |
| **轮子Z偏移** | -0.0805 | m | 高度位置 |

### 万向轮

| 参数 | 值 | 单位 |
|------|-----|------|
| **万向轮半径** | 0.03 | m |
| **万向轮质量** | 0.1 | kg |

### 机器人布局

```
        前方
         ↑
    [左轮]   [右轮]
    ←-- 34.2cm --→

    主体: Φ40.6cm × 21cm
```

---

## ⚙️ 执行器参数

### 驱动系统

| 参数 | 值 | 说明 |
|------|-----|------|
| **驱动类型** | 差速驱动 | Differential Drive |
| **电机类型** | 有刷编码马达 | Brushed Encoder Motor |

### 编码器参数

| 参数 | 值 | 单位 |
|------|-----|------|
| **编码器分辨率** | 1200 | ticks/转 |
| **减速比** | 1.0 | - |
| **电机反转** | False | - |

### Isaac Lab仿真参数

| 参数 | 值 | 单位 | 说明 |
|------|-----|------|------|
| **刚度系数**（stiffness） | 0.0 | N·m/rad | 速度控制模式 |
| **阻尼系数**（damping） | 5.0 | N·m·s/rad | 官方推荐5-20 |
| **力矩上限**（effort_limit） | 20.0 | N·m | 留安全裕度 |
| **速度上限**（velocity_limit） | 5.0 | rad/s | ≈0.32 m/s |

### ROS PID控制参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **Kp**（比例增益） | 50 | 位置误差增益 |
| **Kd**（微分增益） | 20 | 速度误差增益 |
| **Ki**（积分增益） | 0 | 累积误差增益（未使用） |
| **Ko**（前馈增益） | 50 | 输出缩放 |

**控制律**:
```
output = Kp * error_pos + Kd * error_vel + Ko * command
```

---

## 🏃 运动参数

### 速度限制

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **最大线速度**（max_vel_x） | 0.3 | m/s | ROS配置 |
| **最小线速度**（min_vel_x） | -0.3 | m/s | ROS配置（倒车） |
| **最大角速度**（max_vel_theta） | 1.0 | rad/s | ROS配置 |
| **最小角速度**（min_vel_theta） | -1.0 | rad/s | ROS配置 |
| **原地旋转速度** | 0.4 | rad/s | ROS配置 |

### 加速度限制

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **最大线加速度**（acc_lim_x） | 1.0 | m/s² | ROS配置 |
| **最大角加速度**（acc_lim_theta） | 0.6 | rad/s² | ROS配置 |

### 控制频率

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **ROS base_controller** | 10 | Hz | ROS配置 |
| **ROS serial_rate** | 50 | Hz | ROS配置 |
| **Isaac Lab control_dt** | 0.1 | s | 仿真配置 |
| **Isaac Lab sim_dt** | 0.005 | s | 仿真配置（200Hz） |

### 物理仿真参数

| 参数 | 值 | 单位 | 说明 |
|------|-----|------|------|
| **线性阻尼**（linear_damping） | 0.1 | - | 速度衰减 |
| **角阻尼**（angular_damping） | 0.1 | - | 角速度衰减 |
| **物理上线速度上限** | 10.0 | m/s | 仿真器限制 |
| **物理上角速度上限** | 10.0 | rad/s | 仿真器限制 |

### 运动性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **最大直线速度** | 0.3 m/s | 约1.08 km/h |
| **最大旋转速度** | 1.0 rad/s | 约57°/s |
| **最小转弯半径** | 0 m | 原地旋转 |
| **加速度** | 1.0 m/s² | 0到0.3m/s需0.3s |

---

## 📡 传感器参数

### EAI F4 Flash LiDAR（YDLIDAR G4）

#### 基本信息

| 参数 | 值 | 说明 |
|------|-----|------|
| **型号** | EAI F4 Flash | 主型号 |
| **别名** | YDLIDAR G4 | 国际型号 |

#### 扫描参数

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **扫描视野** | 360 | 度 | 全方位 |
| **实物最大距离** | 16.0 | m | 官方规格 |
| **仿真最大距离** | 12.0 | m | RayCaster配置 |
| **最小距离** | 0.15 | m | 盲区 |

#### 频率参数

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **最小扫描频率** | 5.0 | Hz | 官方规格 |
| **最大扫描频率** | 12.0 | Hz | 官方规格 |
| **默认扫描频率** | 7.0 | Hz | 出厂设置 |
| **采样速率** | 9000.0 | Hz | 采样率 |

#### 数据输出

| 参数 | 值 | 说明 |
|------|-----|------|
| **实物数据点数** | 720 点/圈 | 典型2D LiDAR |
| **v6.0仿真射线** | 360 射线 | 优化后配置 |
| **角度分辨率** | 0.5 度 | 720点/360° |

#### 仿真配置

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **仿真更新频率** | 10.0 | Hz | update_period=0.1s |
| **安装高度** | 0.13 | m | 对齐实物 |
| **降采样扇区数** | 36 | - | 每10°一个扇区 |
| **水平分辨率** | 1.0 | 度 | v6.0配置 |

#### 障碍物检测（ROS配置）

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **障碍物检测距离** | 3.0 | m | costmap配置 |
| **障碍物清除距离** | 3.5 | m | costmap配置 |

### LiDAR降采样逻辑

#### 实物处理
```
实物输出: 720点
↓ 重塑为36扇区
每扇区: 20点取最小值
↓
输出: 36维数据
```

#### 仿真处理（v6.0）
```
RayCaster: 360射线
↓ 重塑为36扇区
每扇区: 10点取最小值
↓
输出: 36维数据
```

#### 关键特性
- **扇区角度**: 360° ÷ 36 = 10°/扇区
- **实物每扇区点数**: 720 ÷ 36 = 20点
- **仿真v6.0每扇区点数**: 360 ÷ 36 = 10点
- **归一化范围**: [0, 1]，max_range=12m

---

## 🔌 通信参数

### 串口配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **主串口** | /dev/ttyUSB0 | 默认设备 |
| **备用串口** | /dev/dashgo | 备用路径 |
| **波特率** | 115200 | bps |
| **超时时间** | 0.1 | s |
| **通信频率** | 50 | Hz |
| **传感器状态频率** | 10 | Hz |

---

## 🔋 电源系统

### 电池参数

| 参数 | 值 | 单位 | 说明 |
|------|-----|------|------|
| **标称电压** | 12.0 | V | 典型配置 |
| **电池容量** | 10.0 | Ah | 典型配置 |
| **电机峰值功率** | 20.0 | W | 估算 |

### 续航估算

| 指标 | 值 | 说明 |
|------|-----|------|
| **平坦地面续航** | 2.0 | 小时 |
| **续航里程** | 2.0 | km（以0.3m/s速度） |
| **电池总能量** | 120 | Wh（12V × 10Ah） |

---

## 🧭 导航参数

### 目标公差

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **位置公差**（xy_goal_tolerance） | 0.2 | m | ROS导航配置 |
| **姿态公差**（yaw_goal_tolerance） | 0.1 | rad | ROS导航配置 |

### 路径规划参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **前瞻时间**（sim_time） | 0.6 | s |
| **线速度采样数**（vx_samples） | 12 | - |
| **角速度采样数**（vtheta_samples） | 15 | - |

---

## 🧮 差速驱动运动学

### 速度转换公式

#### 从线速度和角速度到轮速

```python
v_left = (v - w * track_width / 2) / wheel_radius
v_right = (v + w * track_width / 2) / wheel_radius
```

**参数值**:
```python
v = 0.3                # 线速度 (m/s)
w = 1.0                # 角速度 (rad/s)
track_width = 0.342     # 轮距 (m)
wheel_radius = 0.0632   # 轮半径 (m)

# 示例1: 直线前进 (v=0.3, w=0)
v_left = (0.3 - 0 * 0.342 / 2) / 0.0632 = 4.75 rad/s
v_right = (0.3 + 0 * 0.342 / 2) / 0.0632 = 4.75 rad/s

# 示例2: 原地旋转 (v=0, w=1.0)
v_left = (0 - 1.0 * 0.342 / 2) / 0.0632 = -2.70 rad/s
v_right = (0 + 1.0 * 0.342 / 2) / 0.0632 = 2.70 rad/s
```

#### 从轮速到线速度和角速度

```python
v = (v_right + v_left) * wheel_radius / 2
w = (v_right - v_left) * wheel_radius / track_width
```

### 运动学约束

#### 速度约束
```python
# 线速度约束
-max_lin_vel ≤ v ≤ max_lin_vel
-0.3 ≤ v ≤ 0.3  # m/s

# 角速度约束
-max_ang_vel ≤ w ≤ max_ang_vel
-1.0 ≤ w ≤ 1.0  # rad/s
```

#### 轮速约束
```python
# 考虑轮速限制后的有效线速度
v_max_effective = min(max_lin_vel, wheel_radius * velocity_limit)
v_max_effective = min(0.3, 0.0632 * 5.0) = 0.3  # m/s
```

---

## 💻 配置使用方法

### 1. 导入配置

```python
# 方式1: 导入完整配置类
from dashgo_config import DashGoRobotConfig

# 方式2: 导入向后兼容类
from dashgo_config import DashGoROSParams

# 方式3: 导入常量配置
from dashgo_config import MOTION_CONFIG, LIDAR_CONFIG, NAVIGATION_CONFIG
```

### 2. 从YAML加载参数

```python
# 使用完整配置类
config = DashGoRobotConfig.from_yaml()

# 访问物理参数
print(config.physical.wheel_diameter)  # 0.1264
print(config.physical.wheel_radius)    # 0.0632
print(config.physical.wheel_track)     # 0.342

# 访问运动参数
print(config.motion.max_lin_vel)       # 0.3
print(config.motion.max_ang_vel)       # 1.0

# 访问LiDAR参数
print(config.lidar.max_range_real)     # 16.0
print(config.lidar.sim_channels_v6)    # 360
```

### 3. 使用向后兼容类

```python
# 旧代码兼容
params = DashGoROSParams.from_yaml()

# 访问参数（与旧API一致）
print(params.wheel_diameter)           # 0.1264
print(params.wheel_radius)             # 0.0632
print(params.wheel_track)              # 0.342
print(params.encoder_resolution)       # 1200
```

### 4. 使用常量配置

```python
# 在奖励函数中使用
from dashgo_config import MOTION_CONFIG, REWARD_CONFIG

max_vel = MOTION_CONFIG["max_lin_vel"]          # 0.3
high_speed_threshold = REWARD_CONFIG["high_speed_threshold"]  # 0.25

# 速度奖励示例
if current_vel > high_speed_threshold:
    reward += 1.0
```

### 5. 生成配置摘要

```python
config = DashGoRobotConfig.from_yaml()
print(config.summary())
```

**输出示例**:
```
================================================================================
DashGo D1 机器人配置摘要
================================================================================
物理尺寸:
  - 主体直径: 40.6 cm
  - 主体高度: 21.0 cm
  - 主体质量: 13.7 kg
  - 轮子直径: 12.6 cm
  - 轮距: 34.2 cm

运动参数:
  - 最大线速度: 0.3 m/s
  - 最大角速度: 1.0 rad/s
  - 最大线加速度: 1.0 m/s²
  - 最大角加速度: 0.6 rad/s²

LiDAR传感器 (EAI F4 Flash):
  - 扫描范围: 360.0°
  - 最大距离: 16.0 m (实物), 12.0 m (仿真)
  - 扫描频率: 5.0-12.0 Hz
  - 数据点数: 720 点/圈 (实物)
  - 仿真射线: 360 射线 (v6.0)

执行器:
  - 编码器: 1200 ticks/转
  - PID: Kp=50, Kd=20, Ki=0, Ko=50

通信:
  - 串口: /dev/ttyUSB0
  - 波特率: 115200 bps
  - 频率: 50.0 Hz

电源:
  - 电压: 12.0 V
  - 容量: 10.0 Ah
  - 续航: 2.0 小时 (平坦地面)

================================================================================
```

### 6. 在训练环境中使用

```python
# dashgo_env_v2.py 中使用

from dashgo_config import DashGoROSParams, MOTION_CONFIG, LIDAR_CONFIG

class DashGoEnv:
    def __init__(self):
        # 加载ROS参数
        ros_params = DashGoROSParams.from_yaml()
        self.wheel_radius = ros_params.wheel_radius
        self.track_width = ros_params.wheel_track

        # 使用常量配置
        self.max_lin_vel = MOTION_CONFIG["max_lin_vel"]
        self.max_ang_vel = MOTION_CONFIG["max_ang_vel"]
        self.lidar_max_range = LIDAR_CONFIG["max_range"]
```

---

## 📊 参数快速参考表

### 常用参数速查

| 类别 | 参数 | 值 | 单位 |
|------|------|-----|------|
| **物理** | 轮子直径 | 0.1264 | m |
| | 轮距 | 0.342 | m |
| | 机器人半径 | 0.2 | m |
| **运动** | 最大线速度 | 0.3 | m/s |
| | 最大角速度 | 1.0 | rad/s |
| | 线加速度 | 1.0 | m/s² |
| **LiDAR** | 扫描范围 | 360 | 度 |
| | 最大距离（实物） | 16.0 | m |
| | 最大距离（仿真） | 12.0 | m |
| | 扫描频率 | 5-12 | Hz |
| **执行器** | 编码器 | 1200 | ticks/转 |
| | PID Kp | 50 | - |
| | PID Kd | 20 | - |
| **通信** | 波特率 | 115200 | bps |
| | 通信频率 | 50 | Hz |
| **电源** | 电压 | 12 | V |
| | 容量 | 10 | Ah |

---

## ⚠️ 参数注意事项

### 1. 轮子直径差异

**不同批次轮子直径**:
- 0.1212m（旧批次）
- 0.125m（中间批次）
- 0.1264m（当前批次，仿真使用）

### 2. 轮距差异

**不同安装位置轮距**:
- 0.335m（紧凑安装）
- 0.342m（标准安装，仿真使用）
- 0.355m（宽松安装）

### 3. 仿真与实物差异

**常见差异来源**:
- 轮子打滑（真实环境）
- 地面摩擦不均匀
- 电机响应延迟
- 电池电压下降

**建议**:
- 定期标定
- 添加安全余量
- 使用Sim2Real技术（观测噪声、物理随机化）

---

## 📚 相关文档

### ROS配置文件

- `dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml` - 机器人参数
- `dashgo/EAI驱动/dashgo_bringup/config/base_local_planner_params.yaml` - 导航参数
- `dashgo/1/1/nav/param/costmap_common_params.yaml` - 代价地图

### Isaac Lab配置文件

- `dashgo_assets.py` - 机器人资产定义
- `dashgo_config.py` - 本配置文件
- `dashgo_env_v2.py` - 仿真环境

### 参考文档

- `docs/dashgo-robot-specifications_2026-01-24.md` - 机器人规格说明
- `docs/报告1_DashGo文件夹内容分析.md` - ROS参数分析

---

## 📝 版本历史

### v2.0 (2026-01-26)

**重大更新**:
- ✅ 扩展为完整配置系统（7个专门类）
- ✅ 添加所有物理尺寸参数
- ✅ 添加完整LiDAR传感器规格
- ✅ 添加通信、电源、导航参数
- ✅ 添加常量配置字典
- ✅ 添加配置摘要功能
- ✅ 保持向后兼容性

### v1.0 (2026-01-24)

**初始版本**:
- 基础物理参数
- 轮子直径、轮距、编码器

---

**维护者**: Claude Code AI Assistant
**最后更新**: 2026-01-26
**版本**: v2.0
