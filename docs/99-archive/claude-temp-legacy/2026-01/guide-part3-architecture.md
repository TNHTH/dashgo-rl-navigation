# DashGo RL Navigation 项目完全复现指南 v5.0

> **创建时间**: 2026-01-28 22:35:00
> **第三部分**: 项目架构深度解析
> **预计时间**: 20-30分钟
> **依赖**: 第二部分（环境搭建）已完成

---

## 3.1 目录结构详解

### 完整目录树

```
dashgo_rl_project/                         # 项目根目录
│
├── 📄 核心训练文件
│   ├── train_v2.py                        # ⭐ 训练脚本主入口
│   ├── train_cfg_v2.yaml                  # ⭐ 训练超参数配置
│   ├── play.py                            # 演示脚本（可视化训练效果）
│   └── export_torchscript.py              # 模型导出（TorchScript）
│
├── 🤖 环境与资产定义
│   ├── dashgo_env_v2.py                   # ⭐ 仿真环境定义（奖励、传感器、Episode）
│   ├── dashgo_assets.py                   # ⭐ 机器人资产配置（URDF、执行器）
│   └── dashgo_config.py                   # ⭐ ROS参数对齐（Sim2Real）
│
├── 🧠 神经网络架构
│   └── geo_nav_policy.py                  # ⭐ 轻量级网络（v3.1梯度防护）
│
├── 🚀 部署相关
│   ├── scripts/
│   │   ├── geo_distill_node.py            # ROS导航节点
│   │   ├── safety_filter.py               # 安全过滤器
│   │   └── policy_v2.pt                   # 导出的TorchScript模型
│
├── 📚 项目文档
│   ├── README.md                          # 项目总览（644行）
│   └── docs/                              # 分类文档
│       ├── 01-部署指南/
│       ├── 02-训练方案/
│       ├── 03-问题分析/
│       ├── 04-技术规范/
│       ├── 05-协议规范/
│       └── 06-项目历史/
│
├── 🔧 问题记录
│   └── issues/                            # 70+问题记录（按时间排序）
│       ├── 2026-01-27_1730_梯度爆炸导致NaN错误.md
│       ├── 2026-01-27_1727_lidar_sensor实体不存在.md
│       └── ...
│
├── 📋 开发规则
│   └── .agent-workspace/
│       └── rules/
│           ├── isaac-lab-development-iron-rules.md    # ⭐ Isaac Lab 5条铁律
│           ├── project-specific-rules.md              # ⭐ 项目特定规则
│           └── dynamic_rules.md                        # 23条动态规则
│
└── 📁 Sim2Real参数源（只读，严禁修改）
    └── dashgo/                             # 实物ROS包
        └── EAI驱动/
            └── dashgo_bringup/config/
                ├── my_dashgo_params.yaml            # ⭐ 轮径、轮距等物理参数
                └── base_local_planner_params.yaml   # ⭐ 速度限制参数
```

### 关键文件优先级

| 优先级 | 文件 | 用途 | 大小 |
|--------|------|------|------|
| ⭐⭐⭐ | train_v2.py | 训练主入口 | 14.8KB |
| ⭐⭐⭐ | dashgo_env_v2.py | 环境定义（奖励、传感器） | 67.1KB |
| ⭐⭐⭐ | geo_nav_policy.py | 神经网络架构（v3.1） | 8KB |
| ⭐⭐⭐ | train_cfg_v2.yaml | 训练超参数 | 2KB |
| ⭐⭐⭐ | dashgo_config.py | ROS参数对齐 | 17.3KB |
| ⭐⭐ | dashgo_assets.py | 机器人资产配置 | 5KB |
| ⭐⭐ | my_dashgo_params.yaml | 实物物理参数（只读） | 1KB |

---

## 3.2 核心代码分析

### 3.2.1 train_v2.py - 训练脚本主入口

**文件位置**: `train_v2.py`
**文件大小**: 14.8KB
**核心功能**: 启动训练、加载环境、配置RSL-RL Runner

#### 关键代码片段解析

**片段1: AppLauncher初始化（必须最先）**

```python
# 第18-25行
from omni.isaac.lab.app import AppLauncher  # ⚠️ 必须最先导入

# 创建参数解析器
parser = argparse.ArgumentParser()
# 添加--headless参数（无GUI模式）
parser.add_argument("--headless", action="store_true", help="Force display off at startup.")
args_cli = parser.parse_args()

# 启动AppLauncher（必须在使用Isaac Lab之前）
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app  # 获取仿真应用实例
```

**为什么要这样写？**
- Isaac Sim基于Omniverse Kit，必须先启动仿真应用
- 如果先导入`omni.isaac.lab`，headless参数会失效
- 这是Isaac Lab开发铁律第一条

**片段2: 自定义网络注入（关键技术）**

```python
# 第45-52行
def inject_geo_nav_policy():
    """
    注入自定义网络到RSL-RL

    问题：RSL-RL使用eval("GeoNavPolicy")动态加载网络
    解决：在rsl_rl模块中设置GeoNavPolicy属性
    """
    import rsl_rl.runners.on_policy_runner as runner_module
    from geo_nav_policy import GeoNavPolicy  # 导入自定义网络
    setattr(runner_module, "GeoNavPolicy", GeoNavPolicy)  # 注入到RSL-RL模块

# 在创建Runner之前注入
inject_geo_nav_policy()
```

**为什么要注入？**
- RSL-RL的配置文件使用字符串指定网络类名
- RSL-RL使用`eval("GeoNavPolicy")`动态加载
- 必须在`rsl_rl`模块中设置`GeoNavPolicy`属性

**片段3: 环境创建（RSL-RL格式）**

```python
# 第60-75行
# 从配置文件创建环境
env_cfg = DashgoNavEnvV2Cfg()
env_cfg.scene.num_envs = args_cli.num_envs  # 设置并行环境数量

# 创建Isaac Lab环境
env = ManagerBasedRLEnv(cfg=env_cfg)

# 包装为RSL-RL格式（关键步骤）
env = RslRlVecEnvWrapper(env)  # 转换为RSL-RL需要的接口
```

**RslRlVecEnvWrapper的作用**：
- 转换Isaac Lab环境为RSL-RL格式
- 提供Tens orDict格式的观测和奖励
- 支持GPU并行训练

**片段4: 训练器创建与启动**

```python
# 第80-95行
# 加载训练配置
agent_cfg = OmegaConf.load("train_cfg_v2.yaml")

# ⚠️ 配置扁平化处理（RSL-RL要求）
if "runner" in agent_cfg:
    runner_cfg = agent_cfg.pop("runner")  # 提取runner配置
    agent_cfg.update(runner_cfg)          # 合并到根目录

# 创建训练日志目录
log_dir = os.path.join("logs", args_cli.exp_name)
os.makedirs(log_dir, exist_ok=True)

# 创建PPO训练器
runner = OnPolicyRunner(
    env,                    # RSL-RL格式环境
    agent_cfg,              # 扁平化配置
    log_dir=log_dir,        # 日志目录
    device=args_cli.device  # 训练设备（cuda:0或cpu）
)

# 开始训练
runner.learn(num_learning_iterations=agent_cfg.get("max_iterations", 1500))
```

---

### 3.2.2 dashgo_env_v2.py - 仿真环境定义

**文件位置**: `dashgo_env_v2.py`
**文件大小**: 67.1KB（最大的文件）
**核心功能**: 定义机器人环境、奖励函数、传感器、Episode终止条件

#### 环境配置类

```python
# 第35-80行
class DashgoNavEnvV2Cfg(ManagerBasedRLEnvCfg):
    """DashGo导航环境配置"""

    def __init__(self):
        super().__init__()

        # === 场景配置 ===
        self.scene.num_envs = 64          # RTX 4060安全值（不要超过128）
        self.scene.env_spacing = 2.0      # 环境间距（防止机器人互相干扰）
        self.sim.dt = 0.1                 # 仿真时间步长（秒）
        self.scene.episode_length_s = 20.0  # Episode时长（秒）

        # === 机器人配置 ===
        self.robot = DASHGO_D1_CFG        # 机器人资产配置

        # === 传感器配置 ===
        self.sensors = {
            "policy": SensorGroupCfg(
                sensors=[
                    # 4向深度相机（融合成LiDAR）
                    CameraCfg(
                        prim_path="/World/DashGo_D1/chassis_camera_front",
                        update_period=0.1,
                        height=64,
                        width=64,
                        data_type="distance_to_image_plane",
                        attach_debug_visualiz=False,
                    ),
                    # 后、左、右相机（类似配置）
                    # ...
                ]
            )
        }

        # === 奖励配置 ===
        self.rewards = reward_navigation_sota()  # SOTA导航奖励

        # === 课程学习配置 ===
        self._curriculum = CurriculumCfg(...)
```

#### 4相机LiDAR融合实现

```python
# 第150-200行
def process_stitched_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    4向相机融合实现360°LiDAR感知

    输入: 前、后、左、右4个相机
    输出: 72点降采样LiDAR

    为什么是4相机？单相机视野有限，无法实现360°感知
    为什么降采样？原始360点→72点（每5点取1），对齐EAI F4实物雷达
    """
    # 获取前视相机数据 [N, H, W]
    d_front = env.scene["camera_front"].data.output["distance_to_image_plane"]

    # 转换为扫描数据（检测是否有障碍物）
    scan_front = torch.any(d_front > 0, dim=1)  # [N, W]

    # 同样处理其他三个方向
    scan_left = torch.any(env.scene["camera_left"].data.output["distance_to_image_plane"] > 0, dim=1)
    scan_back = torch.any(env.scene["camera_back"].data.output["distance_to_image_plane"] > 0, dim=1)
    scan_right = torch.any(env.scene["camera_right"].data.output["distance_to_image_plane"] > 0, dim=1)

    # 拼接成完整360°扫描 [N, 360]
    full_scan = torch.cat([scan_front, scan_left, scan_back, scan_right], dim=1)

    # 降采样：360点→72点（每5点取1）
    downsampled = full_scan[:, ::5]  # [N, 72]

    # 归一化到[0,1]
    max_range = 5.0  # 最大感知距离（米）
    return downsampled / max_range
```

#### 奖励函数设计（v5.0 Ultimate）

```python
# 第250-300行
def reward_navigation_sota(env: ManagerBasedRLEnv) -> RewardTermCfg:
    """
    SOTA导航奖励函数

    核心思想：
    1. 到达目标奖励（绝对主导）→ 引导机器人完成任务
    2. 进步奖励（保底）→ 确保收敛
    3. 平滑控制奖励 → 避免抖动
    4. 碰撞惩罚 → 安全约束
    """
    return RewardTermCfg(
        func=reward_func,  # 奖励计算函数
        weight={
            # 1. 到达目标（绝对主导）
            "reach_goal": 2000.0,  # 到达目标时给予巨额奖励

            # 2. 进步奖励（保底收敛）
            "progress_to_goal": 1.0,  # 每靠近一点目标就给小奖励

            # 3. 平滑控制（避免抖动）
            "smooth_control": 0.01,  # 速度变化越小越好

            # 4. 碰撞惩罚（安全约束）
            "collision": -50.0,  # 碰撞时给予巨额惩罚

            # 5. 课程学习（自动扩展目标范围）
            "shaping_distance": 0.75,  # v5.0黄金平衡点
        }
    )
```

**为什么这样设计？**
- **稀疏主导（reach_goal=2000）**：鼓励完成任务
- **密集保底（progress=1.0）**：确保训练不会卡死
- **黄金比例（2000:1）**：稀疏奖励是密集奖励的2000倍

---

### 3.2.3 geo_nav_policy.py - 轻量级网络架构v3.1

**文件位置**: `geo_nav_policy.py`
**文件大小**: 8KB
**核心功能**: 定义神经网络结构（Actor-Critic）

#### 网络架构概览

```
输入: 246维观测
├── LiDAR: 216维（3帧历史堆叠）
└── 状态: 30维（位置、速度、目标等）

    ↓

GeoNavPolicy (v3.1)
├── geo_encoder (1D-CNN)
│   ├── Conv1D(1→16) + LayerNorm + ELU
│   ├── Conv1D(16→32) + LayerNorm + ELU
│   └── Linear(32*54→64) + LayerNorm + ELU
│
├── fusion_layer
│   └── Linear(64+30→128) + LayerNorm + ELU
│
├── actor_head
│   └── Linear(128→2)  # 输出线速度和角速度
│
└── critic_head
    └── Linear(128→1)  # 输出价值估计

    ↓

输出: 2维动作
├── action[0]: 线速度 (m/s)
└── action[1]: 角速度 (rad/s)
```

#### v3.1梯度爆炸防护

```python
# 第50-80行
class GeoNavPolicy(nn.Module):
    """轻量级导航网络（v3.1梯度防护）"""

    def __init__(self, ...):
        super().__init__()

        # === 几何特征编码器（1D-CNN）===
        self.geo_encoder = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([16, 108]),  # ⭐ v3.1: 添加LayerNorm防止梯度爆炸
            nn.ELU(),

            # 第二层卷积
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([32, 54]),   # ⭐ v3.1: 添加LayerNorm
            nn.ELU(),

            # 展平并降维
            nn.Flatten(),
            nn.Linear(32 * 54, 64),
            nn.LayerNorm(64),         # ⭐ v3.1: 添加LayerNorm
            nn.ELU()
        )

        # ⭐ v3.1: 输入裁剪（防止梯度爆炸）
        self.input_clamp = ClampModule(min_val=-5.0, max_val=5.0)

        # === 特征融合层 ===
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 30, 128),  # 64(CNN) + 30(state)
            nn.LayerNorm(128),        # ⭐ v3.1: 添加LayerNorm
            nn.ELU()
        )

        # === Actor（策略网络）===
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),         # ⭐ v3.1: 添加LayerNorm
            nn.ELU(),
            nn.Linear(64, 2)          # 输出2维动作（线速度、角速度）
        )

        # === Critic（价值网络）===
        self.critic_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),        # ⭐ v3.1: 添加LayerNorm
            nn.ELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),         # ⭐ v3.1: 添加LayerNorm
            nn.ELU(),
            nn.Linear(64, 1)          # 输出价值估计
        )

        # ⭐ v3.1: 正交初始化（防止梯度爆炸）
        self.apply(orthogonal_init)
```

**v3.1的三重防护**：
1. **LayerNorm**：归一化每一层输出
2. **Input Clamp**：裁剪输入范围
3. **Orthogonal Init**：正交初始化权重

---

### 3.2.4 dashgo_config.py - ROS参数对齐

**文件位置**: `dashgo_config.py`
**文件大小**: 17.3KB
**核心功能**: 从ROS配置读取实物参数，实现Sim2Real对齐

#### 关键参数读取

```python
# 第30-60行
class DashGoROSParams:
    """从ROS配置读取参数"""

    @staticmethod
    def from_yaml(yaml_path="dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml"):
        """读取ROS YAML配置"""
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)

        return DashGoROSParams(
            # 物理参数（精确到0.0001米）
            wheel_diameter=params["wheel_diameter"],  # 0.1264 m
            wheel_radius=params["wheel_diameter"] / 2,  # 0.0632 m
            wheel_track=params["wheel_track"],    # 0.3420 m

            # 速度限制
            max_lin_vel=params["max_vel_x"],      # 0.3 m/s
            max_ang_vel=params["max_rot_vel"],    # 1.0 rad/s
        )

# 使用示例
ros_params = DashGoROSParams.from_yaml()
print(f"轮子半径: {ros_params.wheel_radius} m")  # 0.0632 m（精确）
```

**为什么要精确到0.0001？**
- 1%的轮径误差 = 10cm定位误差（累积10米后）
- Sim2Real对齐的关键：仿真参数必须精确对齐实物

---

## 3.3 数据流图

### 完整训练数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     训练循环（每iteration）                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  1. 环境重置     │  env.reset()
│  - 随机目标位置   │
│  - 机器人初始位置 │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  2. 收集数据     │  for step in range(num_steps_per_env):
│  - 策略推理      │      actions = policy.act(obs)
│  - 环境交互      │      next_obs, rewards, dones = env.step(actions)
│  - 存储经验      │      buffer.add(obs, actions, rewards, dones)
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  3. PPO更新     │  runner.learn()
│  - 计算优势      │      advantages = compute_gae()
│  - 策略梯度      │      policy_loss = compute_ppo_loss()
│  - 价值函数更新  │      value_loss = compute_value_loss()
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  4. 保存Checkpoint │  if iteration % save_interval == 0:
│  - 模型权重      │      torch.save(model.state_dict(), ...)
│  - 训练日志      │      writer.add_scalar(...)
└─────────────────┘
```

### 单个Episode数据流

```
Episode开始 (env.reset())
    │
    ▼
┌─────────────────┐
│  观测环境        │
│  - LiDAR (72维) │
│  - 目标向量 (3维)│
│  - 速度 (2维)   │
│  → obs (246维)  │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  策略推理        │
│  action = policy(obs)
│  → action (2维)  │
│  [v, w]         │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  执行动作        │
│  env.step(action)│
│  - 控制机器人运动 │
│  - 仿真物理更新  │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  计算奖励        │
│  reward = compute_reward()
│  - reach_goal   │
│  - progress     │
│  - collision    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  检查终止        │
│  done = check_done()
│  - 到达目标？   │
│  - 碰撞？       │
│  - 超时？       │
└────────┬─────────┘
         │
         ▼
    如果done → Episode结束
    否则 → 继续下一step
```

---

## 3.4 模块依赖关系

### 核心依赖图

```
train_v2.py (训练入口)
    │
    ├── AppLauncher (Isaac Sim启动)
    │   └── omni.isaac.lab.app
    │
    ├── DashgoNavEnvV2Cfg (环境配置)
    │   ├── DASHGO_D1_CFG (机器人资产)
    │   │   └── dashgo_assets.py
    │   │
    │   ├── reward_navigation_sota (奖励函数)
    │   │   └── dashgo_env_v2.py
    │   │
    │   └── process_stitched_lidar (传感器处理)
    │       └── dashgo_env_v2.py
    │
    ├── ManagerBasedRLEnv (Isaac Lab环境)
    │   └── omni.isaac.lab.envs
    │
    ├── RslRlVecEnvWrapper (RSL-RL包装)
    │   └── isaaclab_rl.rsl_rl
    │
    ├── GeoNavPolicy (神经网络)
    │   ├── geo_nav_policy.py
    │   └── torch.nn
    │
    ├── OnPolicyRunner (PPO训练器)
    │   ├── rsl_rl.runners
    │   └── train_cfg_v2.yaml (配置)
    │
    └── DashGoROSParams (参数对齐)
        └── dashgo_config.py
            └── dashgo/ (ROS配置，只读)
```

### 导入顺序（铁律）

```python
# ✅ 正确顺序
1. import argparse
2. from omni.isaac.lab.app import AppLauncher  # 必须最先
3. parser = argparse.ArgumentParser()
4. app_launcher = AppLauncher(headless=args.headless)
5. simulation_app = app_launcher.app
6. # 然后才能导入其他库
7. import torch
8. import gymnasium as gym
9. from omni.isaac.lab.envs import ManagerBasedRLEnv
10. from rsl_rl.runners import OnPolicyRunner

# ❌ 错误顺序
1. import torch  # 错误！太早了
2. from omni.isaac.lab.envs import ManagerBasedRLEnv  # 错误！太早了
3. from omni.isaac.lab.app import AppLauncher  # 太晚了
```

---

## 3.5 关键实现细节

### 3.5.1 动作空间设计

**连续动作空间**：
```python
# 输出: [batch, 2]
# action[0]: 线速度 (m/s) ∈ [-0.3, 0.3]
# action[1]: 角速度 (rad/s) ∈ [-1.0, 1.0]

# 硬裁剪到实物限制
max_lin_vel = 0.3  # m/s
max_ang_vel = 1.0  # rad/s

target_v = torch.clamp(action[:, 0] * max_lin_vel, -max_lin_vel, max_lin_vel)
target_w = torch.clamp(action[:, 1] * max_ang_vel, -max_ang_vel, max_ang_vel)
```

### 3.5.2 观察空间设计

**多模态观测融合**：
```python
# 输入维度: [batch, 246]
# - LiDAR: [batch, 216] (3帧历史堆叠，每帧72维)
# - 状态: [batch, 30] (位置、速度、目标等)

# 关键：历史帧提供短时记忆
lidar_history = [current_lidar, prev_lidar, prev_prev_lidar]
fused_obs = torch.cat(lidar_history, dim=1)  # [batch, 216]
```

### 3.5.3 课程学习系统

**自动自适应课程**：
```python
# v6.0特性：无论num_envs多少，都在75%训练时完成课程
current_num_envs = 64  # 实际环境数
max_iters = 8000       # 总训练轮数
steps_per_env = 24     # 每轮步数

total_steps = current_num_envs * max_iters * steps_per_env
curriculum_end_step = int(total_steps * 0.75)  # 75%进度完成

# 动态调整目标范围
target_range = lerp(0.5, 3.0, current_step / curriculum_end_step)
```

### 3.5.4 Sim2Real对齐策略

**参数精确对齐**：
```python
# 从ROS配置读取真实参数
ros_params = DashGoROSParams.from_yaml()
wheel_radius = ros_params.wheel_radius  # 0.0632 m（精确）

# 仿真中应用
actuators={
    "wheels": ArticulationCfg.ActuatorCfg(
        effort_limit_sim=20.0,  # 对齐实物转矩限制
        velocity_limit_sim=5.0,  # 对齐实物速度限制
        stiffness=0.0,           # 速度控制模式（对齐实物PID）
        damping=5.0,             # 对齐实物阻尼
    )
}
```

---

## 3.6 下一步

**恭喜！** 你已经深入理解了：

✅ 完整目录结构（每个文件的用途）
✅ 核心代码分析（train_v2.py, dashgo_env_v2.py, geo_nav_policy.py）
✅ 数据流图（训练循环、Episode循环）
✅ 模块依赖关系
✅ 关键实现细节（动作空间、观察空间、课程学习、Sim2Real）

**下一部分**：训练实战指南

我们将一起：
- 学习训练前检查清单（5条铁律）
- 理解训练配置详解
- 启动第一次训练（headless模式）
- 监控训练过程（TensorBoard）
- 解决常见训练问题

**预计时间**：15-25分钟

---

**第三部分完成** | 总进度: 43% (3/7)
