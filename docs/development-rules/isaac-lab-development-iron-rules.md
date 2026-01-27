# Isaac Lab 开发铁律

> **版本**: v1.0
> **生效日期**: 2026-01-24
> **适用范围**: DashGo RL Navigation 项目（Isaac Lab + RSL-RL）
> **严重程度**: 🔴 严格强制（违反将导致训练失败或系统崩溃）

---

## 🚨 规则概述

基于 RTX 4060 Laptop (8GB VRAM) 硬件环境和 DashGo D1 项目需求，以下5条规则是**绝对不能触碰的铁律**。

违反这些规则将导致：
- ❌ "代码没逻辑错误，但就是跑不起来"
- ❌ "训练三天模型完全不收敛"
- ❌ "显存溢出、系统崩溃"
- ❌ "Sim2Real完全失败"

---

## 🛑 规则一：Python 导入顺序是"生死线"

### 严重程度
🔴 **最高** - 违反导致参数失效、Segfault、训练无法启动

### 规则内容

**必须最先导入 AppLauncher 并启动仿真应用，然后才能导入其他库。**

### 正确范式

```python
# ✅ 正确顺序
import argparse  # 1. 只有 argparse 可以放在最前面

from omni.isaac.lab.app import AppLauncher  # 2. 必须第二

# 3. 解析参数并启动 APP
parser = argparse.ArgumentParser()
args = parser.parse_args()
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# 4. 只有在这里，才能导入其他库
import torch
import gymnasium as gym
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from rsl_rl.runners import OnPolicyRunner

# ❌ 错误示例：先导入 Isaac Lab 模块
from omni.isaac.lab.envs import ManagerBasedRLEnv  # 错误！太早了
from omni.isaac.lab.app import AppLauncher  # 太晚了，headless 会失效
```

### 原因说明

Isaac Sim 基于 Omniverse Kit 构建。如果先导入 `omni.isaac.core`，Python 会尝试在"空壳"环境中寻找仿真器实例，导致：
- `--headless` 参数被忽略
- Segfault（段错误）
- 训练无法启动

### 强制检查清单

- [ ] `AppLauncher` 在第35行之前（所有 Isaac Lab 模块之前）
- [ ] `argparse` 在 `AppLauncher` 之前
- [ ] `simulation_app = app_launcher.app` 在导入 `omni.isaac.*` 之前
- [ ] 训练脚本 `main()` 函数的第一行就是 AppLauncher 初始化

### 违反后果

**历史证据**：
- ❌ `issues/2026-01-24_1726_训练启动失败配置错误与Headless失效.md`
- ❌ `--headless` 参数失效，窗口弹出
- ❌ 训练无法启动

---

## 🧩 规则二：RSL-RL 的配置"洁癖"

### 严重程度
🔴 **高** - 违反导致 KeyError、训练器初始化失败

### 规则内容

**RSL-RL 的 Runner 类只接受"扁平化"的字典，必须预处理 YAML 配置。**

### 强制要求

```python
# ❌ 错误：直接使用嵌套配置
agent_cfg = yaml.safe_load("train_cfg_v2.yaml")
# 结果：{'runner': {'num_steps_per_env': 24, ...}, ...}
runner = OnPolicyRunner(env, agent_cfg)  # ❌ KeyError!

# ✅ 正确：扁平化配置
agent_cfg = yaml.safe_load("train_cfg_v2.yaml")

# [强制] 预处理：提取嵌套参数到根目录
if "runner" in agent_cfg:
    runner_cfg = agent_cfg.pop("runner")
    agent_cfg.update(runner_cfg)  # 把 num_steps_per_env 提到根目录

if "algorithm" in agent_cfg:
    algo_cfg = agent_cfg.pop("algorithm")
    agent_cfg.update(algo_cfg)

if "policy" in agent_cfg:
    policy_cfg = agent_cfg.pop("policy")
    agent_cfg.update(policy_cfg)

runner = OnPolicyRunner(env, agent_cfg)  # ✅ 成功
```

### 原因说明

RSL-RL 的 `OnPolicyRunner.__init__()` 直接访问：
```python
self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 直接在根目录找
```

如果配置嵌套在 `runner` 下，就会报 `KeyError`。

### YAML 文件组织

**推荐结构**（保持可读性 + 兼容性）：
```yaml
# train_cfg_v2.yaml
runner:
  num_steps_per_env: 24  # 可读性好
  max_iterations: 1500

algorithm:
  learning_rate: 1e-3
  clip_param: 0.2

policy:
  actor_hidden_dims: [512, 256, 128]
```

**Python 扁平化**（在脚本中处理）：
```python
# train_v2.py 中自动扁平化
if "runner" in agent_cfg:
    agent_cfg.update(agent_cfg.pop("runner"))
```

### 强制检查清单

- [ ] YAML 中可以有嵌套结构（为了可读性）
- [ ] Python 脚本中必须有扁平化代码
- [ ] 使用 `cfg.pop()` + `cfg.update()` 模式
- [ ] 访问配置时使用 `.get()` 方法（安全）

### 违反后果

**历史证据**：
- ❌ `issues/2026-01-24_1726_训练启动失败配置错误与Headless失效.md`
- ❌ `KeyError: 'num_steps_per_env'`
- ❌ 训练器初始化失败

---

## 💾 规则三：显存管理的"4060 生存法则"

### 严重程度
🔴 **高** - 违反导致 OOM、训练速度崩溃、系统卡死

### 硬件限制

**RTX 4060 Laptop (8GB VRAM)** - 温饱线边缘

### 强制要求

#### 3.1 环境数量限制

```yaml
# ❌ 严禁超过 128
num_envs: 256  # 可能 OOM
num_envs: 512  # 极可能 OOM
num_envs: 1024  # 必定 OOM

# ✅ 推荐值
num_envs: 64   # 保守（稳定）
num_envs: 80   # 平衡
num_envs: 128  # 上限（需要监控显存）
```

**原因**：RSL-RL 宣称支持 4096 环境（指 A100 80GB），不适合 8GB 显存。

#### 3.2 传感器选择

```python
# ❌ 避免使用 Camera（显存杀手）
sensor = CameraCfg(
    prim_path="...",
    # 消耗大量显存用于渲染
)

# ✅ 推荐：RayCaster（高效）
sensor = RayCasterCfg(
    prim_path="...",
    pattern_cfg=patterns.LidarPatternCfg(...),
    # 显存占用低 40%
)
```

**原因**：Camera 渲染极其消耗显存，RayCaster 仅进行射线检测。

#### 3.3 强制清理显存

```python
# [强制] 在训练前必须执行
torch.cuda.empty_cache()

# 然后才能创建 Runner
runner = OnPolicyRunner(env, agent_cfg, ...)
```

### 监控指标

```bash
# 另一个终端监控显存
watch -n 1 nvidia-smi

# 正常范围：
# GPU 利用率：80-95%
# 显存占用：6-7GB（留1-2GB余量）
# 温度：< 80°C
```

### 强制检查清单

- [ ] `num_envs` ≤ 128
- [ ] 使用 RayCaster 传感器（非 Camera）
- [ ] 训练前执行 `torch.cuda.empty_cache()`
- [ ] 监控 `nvidia-smi` 显存占用
- [ ] 如果显存 > 7GB，降低 `num_envs`

### 违反后果

- ⚠️ 显存溢出（OOM）
- ⚠️ 训练速度从 1000 FPS 掉到 0.1 FPS
- ⚠️ 系统卡死或崩溃

---

## 🤖 规则四：物理参数的"实机对齐"

### 严重程度
🔴 **高** - 违反导致 Sim2Real 完全失败（90%原因）

### 强制要求

#### 4.1 轮子参数（硬编码）

```python
# ✅ 必须使用真实参数（从 ROS 配置读取）
wheel_radius = 0.0632  # m（精确到小数点后4位）
wheel_track = 0.3420    # m（精确到小数点后4位）

# ❌ 禁止猜测或使用默认值
wheel_radius = 0.06  # 错误！误差 5%
wheel_track = 0.34    # 错误！误差 0.6%
```

**来源**：
- ROS 配置：`dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml`
- `dashgo_config.py` 中的 `DashGoROSParams` 类

#### 4.2 控制模式配置

```python
# ✅ 正确：速度控制模式
actuators={
    "wheels": ArticulationCfg.ActuatorCfg(
        joint_names_expr=[".*_wheel_joint"],
        stiffness=0.0,        # ✅ 速度控制：刚度为0
        damping=5.0,           # ✅ 主要靠阻尼
        effort_limit_sim=20.0,
        velocity_limit_sim=5.0,
    ),
}

# ❌ 错误：力矩控制模式
stiffness=100.0,  # 错误！实机底层是 PID
# effort控制      # 错误！实机不支持
```

**原因**：实机 DashGo 底层是 PID 控制，仿真必须对齐。

#### 4.3 速度截断

```python
# ✅ 训练输出必须截断
max_lin_vel = 0.3  # m/s（实机最大速度）
max_ang_vel = 1.0  # rad/s（实机最大角速度）

# 在动作处理中硬裁剪
target_v = torch.clamp(actions[:, 0] * max_lin_vel, -max_lin_vel, max_lin_vel)
target_w = torch.clamp(actions[:, 1] * max_ang_vel, -max_ang_vel, max_ang_vel)
```

**原因**：实机无法执行超过 0.3 m/s 的速度。

### 参数来源

**文件**：`docs/dashgo-robot-specifications_2026-01-24.md`

**关键参数**：
```yaml
wheel_diameter: 0.1264 m  # 轮子直径
wheel_radius: 0.0632 m    # 轮子半径
wheel_track: 0.3420 m     # 轮距
max_lin_vel: 0.3 m/s       # 最大线速度
max_ang_vel: 1.0 rad/s     # 最大角速度
```

### 强制检查清单

- [ ] 轮子半径精确到小数点后4位（0.0632）
- [ ] 轮距精确到小数点后4位（0.3420）
- [ ] stiffness = 0.0（速度控制模式）
- [ ] 使用 `DashGoROSParams.from_yaml()` 读取参数
- [ ] 动作输出硬裁断在 ±0.3 m/s
- [ ] 定期对比仿真与实机参数一致性

### 违反后果

- ❌ 里程计误差累积
- ❌ 仿真策略无法部署到实机
- ❌ Sim2Real 完全失败

---

## 🔄 规则五：坐标系的"幽灵"

### 严重程度
🟡 **中** - 违反导致 Episode 瞬间结束、训练无法收敛

### 强制要求

**Isaac Sim 使用 Z-up, Right-handed（Z轴向上，右手系）。**

**USD 资产文件必须确保机器人自然平放在地面。**

### 检查方法

1. **打开 Isaac Sim GUI**
2. **拖入机器人 USD 文件**
3. **检查**：
   - 轮子是否陷在地里？
   - 机器人是否侧躺着？
   - 机器人是否悬空？

### 常见问题

```python
# ❌ 错误：Y-up 坐标系（Blender导出）
# 机器人可能侧躺着或倒着

# ✅ 正确：Z-up 坐标系（Isaac Sim 标准）
# 机器人自然平放
```

### 修复方法

**如果方向不对**：
- ❌ 不要在代码里强行旋转（治标不治本）
- ✅ 修改 USD 文件的 `rootXform`
- ✅ 确保在 (0,0,0) 时机器人自然平放

### 强制检查清单

- [ ] USD 文件在 Isaac Sim GUI 中打开正常
- [ ] 机器人自然平放在地面（无倾斜、无侧翻）
- [ ] 轮子与地面接触（无悬空、无陷入）
- [ ] 机器人朝向正确（前 方指向 +X 或 +Y）

### 违反后果

- ⚠️ Episode 开始瞬间结束（检测到"碰撞"）
- ⚠️ 训练无法收敛（机器人一直"翻车"）
- ⚠️ Reward 持续为负

---

## 📋 开发检查清单

### 训练启动前（必须执行）

- [ ] **规则一检查**：AppLauncher 在所有 Isaac Lab 模块之前
- [ ] **规则二检查**：配置扁平化代码已添加
- [ ] **规则三检查**：`num_envs` ≤ 128，使用 RayCaster
- [ ] **规则四检查**：物理参数从 ROS 配置读取
- [ ] **规则五检查**：USD 文件在 GUI 中验证过

### 训练过程中（持续监控）

- [ ] 显存占用 < 7GB
- [ ] GPU 温度 < 80°C
- [ ] Reward 曲线是否正常（不持续下降）
- [ ] Episode length 是否增长

---

## 🔍 常见错误案例

### 案例1：Headless 失效

**错误现象**：
```bash
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 80
# 结果：窗口仍然弹出
```

**原因**：违反规则一（导入顺序错误）

**修复**：
```python
# 确保这个在最前面
from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
# 然后才能导入其他模块
```

---

### 案例2：训练崩溃 OOM

**错误现象**：
```
RuntimeError: CUDA out of memory
```

**原因**：违反规则三（`num_envs` 太大）

**修复**：
```yaml
num_envs: 64  # 从 256 降低到 64
```

---

### 案例3：Sim2Real 失败

**错误现象**：
- 仿真训练成功率 90%
- 实机测试完全失败（机器人不动、乱动、撞墙）

**原因**：违反规则四（物理参数不准）

**修复**：
```python
# 从 ROS 配置读取真实参数
ros_params = DashGoROSParams.from_yaml()
self.wheel_radius = ros_params.wheel_radius  # 0.0632
self.track_width = ros_params.wheel_track    # 0.3420
```

---

## 📚 相关文档

1. **问题记录**：
   - `issues/2026-01-24_1726_训练启动失败配置错误与Headless失效.md`
   - `issues/2026-01-24_0128_headless相机prim错误.md`

2. **参数规格**：
   - `docs/dashgo-robot-specifications_2026-01-24.md`

3. **架构师建议**：
   - `docs/architect-recommendations/sim2real-deployment-strategy_2026-01-24.md`

---

## 🎯 执行检查清单

### 每次训练前必须执行

- [ ] 规则一：检查导入顺序（AppLauncher 在第35行之前）
- [ ] 规则二：检查配置扁平化代码存在
- [ ] 规则三：检查 `num_envs` ≤ 128，清除显存
- [ ] 规则四：确认物理参数从 ROS 配置读取
- [ ] 规则五：USD 文件已在 GUI 中验证

### 违反规则的后果

| 规则 | 违反后果 | 恢复难度 |
|------|---------|-----------|
| 规则一 | 训练无法启动、headless失效 | 低（修改导入顺序） |
| 规则二 | KeyError、初始化失败 | 低（添加扁平化代码） |
| 规则三 | OOM、速度崩溃、系统卡死 | 中（降低 num_envs） |
| 规则四 | Sim2Real 完全失败 | 高（重新训练） |
| 规则五 | Episode 瞬间结束 | 中（修复 USD 文件） |

---

**维护者**: Claude Code AI Assistant
**最后更新**: 2026-01-24
**版本**: v1.0
**来源**: Isaac Sim Architect（基于 RTX 4060 Laptop + DashGo D1 实战经验）
**状态**: 🔴 严格强制
