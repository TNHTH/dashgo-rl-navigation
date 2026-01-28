# DashGo D1 机器人参数规格

> **创建时间**: 2026-01-24
> **机器人型号**: DashGo D1
> **参数来源**: ROS配置文件 + Isaac Lab仿真配置
> **文档用途**: 统一参数参考，确保仿真与实物对齐

---

## 📐 物理参数

### 驱动系统参数

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **驱动类型** | 差速驱动（Differential Drive） | - | 结构 |
| **轮子直径** | 0.1264 | m | ROS配置 |
| **轮子半径** | 0.0632 | m | 计算（diameter/2） |
| **轮距**（左右轮中心距） | 0.3420 | m | ROS配置 |
| **编码器分辨率** | 1200 | ticks/转 | ROS配置 |
| **减速比** | 1.0 | - | ROS配置 |
| **电机反转** | False | - | ROS配置 |

**轮子布局**：
```
        前方
         ↑
    [左轮]   [右轮]
    ←-- 0.342m --→
```

**说明**：
- 轮子直径根据不同批次有差异（0.1212-0.1264m）
- 当前仿真使用 `0.1264m`（与ROS配置一致）
- 轮距根据安装位置有差异（0.335-0.342m）
- 当前仿真使用 `0.342m`（与ROS配置一致）

---

## ⚙️ 执行器参数（电机）

### 电机驱动参数

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **阻尼系数**（damping） | 5.0 / 10.0 | N·m·s/rad | Isaac Lab / ROS |
| **刚度系数**（stiffness） | 0.0 | N·m/rad | Isaac Lab |
| **力矩上限**（effort_limit） | 20.0 | N·m | Isaac Lab |
| **速度上限**（velocity_limit） | 5.0 | rad/s | Isaac Lab |
| **加速度上限**（accel_limit） | 1.0 | m/s² | ROS配置 |

**Isaac Lab仿真配置**：
```python
# dashgo_assets.py
actuators={
    "dashgo_wheels": ImplicitActuatorCfg(
        joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
        stiffness=0.0,
        damping=5.0,              # 阻尼系数
        effort_limit_sim=20.0,    # 力矩上限
        velocity_limit_sim=5.0,   # 速度上限
    ),
}
```

**ROS配置**：
```yaml
# my_dashgo_params.yaml
accel_limit: 1.0  # 加速度上限
```

---

## 🏃 运动参数

### 速度限制

| 参数 | 值 | 单位 | 来源 |
|------|-----|------|------|
| **最大线速度**（max_lin_vel） | 0.3 | m/s | ROS配置 |
| **最大角速度**（max_ang_vel） | 1.0 | rad/s | ROS配置 |
| **最大线加速度** | 1.0 | m/s² | ROS配置 |
| **最大角加速度** | 0.6 | rad/s² | ROS配置 |
| **控制频率** | 10 / 50 | Hz | ROS / Isaac Lab |

**ROS配置参考**：
```yaml
# 典型ROS导航配置
max_vel_x: 0.3          # 最大线速度
min_vel_x: -0.3         # 最小线速度（倒车）
max_vel_theta: 1.0      # 最大角速度
min_vel_theta: -1.0     # 最小角速度
acc_lim_x: 1.0          # 线加速度上限
acc_lim_theta: 0.6      # 角加速度上限
```

**控制频率对比**：
- ROS base_controller: 10 Hz
- ROS serial_rate: 50 Hz
- Isaac Lab control_dt: 0.1s (10 Hz)
- Isaac Lab sim_dt: 0.005s (200 Hz)

---

## 🎮 PID控制参数

### 速度环PID参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **Kp**（比例系数） | 50 | 位置误差增益 |
| **Kd**（微分系数） | 20 | 速度误差增益 |
| **Ki**（积分系数） | 0 | 累积误差增益（未使用） |
| **Ko**（前馈系数） | 50 | 输出缩放 |

**ROS配置**：
```yaml
# my_dashgo_params.yaml
Kp: 50    # 比例增益
Kd: 20    # 微分增益
Ki: 0     # 积分增益（未使用）
Ko: 50    # 前馈增益
```

**控制律**：
```
output = Kp * error_pos + Kd * error_vel + Ko * command
```

---

## 📡 通信参数

### 串口配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **串口** | /dev/ttyUSB0 或 /dev/dashgo | 设备路径 |
| **波特率** | 115200 | bps |
| **超时时间** | 0.1 | s |
| **通信频率** | 50 | Hz |
| **传感器状态频率** | 10 | Hz |

**ROS配置**：
```yaml
port: /dev/ttyUSB0
baud: 115200
timeout: 0.1
rate: 50
sensorstate_rate: 10
```

---

## 🤖 物理仿真参数

### 刚体属性

| 参数 | 值 | 单位 | 说明 |
|------|-----|------|------|
| **线性阻尼**（linear_damping） | 0.1 | - | 速度衰减 |
| **角阻尼**（angular_damping） | 0.1 | - | 角速度衰减 |
| **最大线速度** | 10.0 | m/s | 物理上限 |
| **最大角速度** | 10.0 | rad/s | 物理上限 |
| **重力** | 启用 | - | disable_gravity=False |

### 关节属性

| 参数 | 值 | 说明 |
|------|-----|------|
| **自碰撞** | 禁用 | enabled_self_collisions=False |
| **位置迭代次数** | 8 | solver_position_iteration_count |
| **速度迭代次数** | 4 | solver_velocity_iteration_count |
| **睡眠阈值** | 0.005 | sleep_threshold |
| **稳定阈值** | 0.001 | stabilization_threshold |

---

## 🧮 差速驱动运动学

### 速度转换公式

**从线速度和角速度到轮速**：
```python
v_left = (v - w * track_width / 2) / wheel_radius
v_right = (v + w * track_width / 2) / wheel_radius
```

**参数值**：
```python
v = 0.3          # 线速度 (m/s)
w = 1.0          # 角速度 (rad/s)
track_width = 0.342  # 轮距 (m)
wheel_radius = 0.0632 # 轮半径 (m)

# 示例计算（v=0.3, w=0，直线前进）
v_left = (0.3 - 0 * 0.342 / 2) / 0.0632 = 4.75 rad/s
v_right = (0.3 + 0 * 0.342 / 2) / 0.0632 = 4.75 rad/s
```

**从轮速到线速度和角速度**：
```python
v = (v_right + v_left) * wheel_radius / 2
w = (v_right - v_left) * wheel_radius / track_width
```

---

## 📊 关键性能指标

### 运动性能

| 指标 | 值 | 说明 |
|------|-----|------|
| **最大直线速度** | 0.3 m/s | 约1.08 km/h |
| **最大旋转速度** | 1.0 rad/s | 约57°/s |
| **最小转弯半径** | 0 m | 原地旋转（差速驱动） |
| **加速度** | 1.0 m/s² | 0到0.3m/s需0.3s |

### 续航估算

| 参数 | 值 | 说明 |
|------|-----|------|
| **电机功率** | ~20W | 峰值 |
| **电池容量** | ~10Ah | 典型配置 |
| **续航时间** | ~2小时 | 平坦地面 |
| **续航里程** | ~2km | 以0.3m/s速度 |

---

## 🔧 参数使用指南

### 仿真环境配置

**在 `dashgo_env_v2.py` 中使用**：
```python
# 从ROS配置加载参数
ros_params = DashGoROSParams.from_yaml()
self.wheel_radius = ros_params.wheel_radius  # 0.0632m
self.track_width = ros_params.wheel_track    # 0.342m

# 运动限制
max_lin_vel = MOTION_CONFIG["max_lin_vel"]   # 0.3 m/s
max_ang_vel = MOTION_CONFIG["max_ang_vel"]   # 1.0 rad/s
```

### 训练超参数配置

**在 `train_cfg_v2.yaml` 中配置**：
```yaml
# 根据机器人速度范围设置奖励阈值
high_speed_threshold: 0.25  # 小于max_lin_vel (0.3)
```

---

## 📝 参数修改记录

### 2026-01-24 - 统一参数管理

**修改**：
- ✅ 创建 `DashGoROSParams` 类统一管理参数
- ✅ 从ROS YAML文件读取参数
- ✅ 添加常量配置字典（MOTION_CONFIG, REWARD_CONFIG等）

**改进**：
- 避免硬编码
- 仿真与实物参数对齐
- 易于维护和调优

---

## ⚠️ 参数注意事项

### 1. 轮子直径差异

**不同批次轮子直径**：
- 0.1212m（旧批次）
- 0.125m（中间批次）
- 0.1264m（当前批次）

**建议**：
- 测量实际轮子直径
- 更新 `my_dashgo_params.yaml`
- 重新标定里程计

### 2. 轮距差异

**不同安装位置轮距**：
- 0.335m（紧凑安装）
- 0.342m（标准安装）
- 0.355m（宽松安装）

**建议**：
- 测量左右轮中心距
- 更新配置文件
- 重新标定旋转角度

### 3. 仿真与实物差异

**常见差异来源**：
- 轮子打滑（真实环境）
- 地面摩擦不均匀
- 电机响应延迟
- 电池电压下降

**建议**：
- 定期标定
- 添加安全余量
- 使用Sim2Real技术（观测噪声、物理随机化）

---

## 📚 参考文档

1. **ROS配置文件**：
   - `dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml`
   - `dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params_fl.yaml`

2. **Isaac Lab配置**：
   - `dashgo_assets.py` - 机器人资产定义
   - `dashgo_config.py` - ROS参数加载
   - `dashgo_env_v2.py` - 仿真环境

3. **Isaac Sim文档**：
   - https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/

---

**维护者**: Claude Code AI Assistant
**最后更新**: 2026-01-24
**版本**: v1.0
