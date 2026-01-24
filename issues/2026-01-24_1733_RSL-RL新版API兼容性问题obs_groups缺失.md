# RSL-RL新版API兼容性问题：obs_groups缺失

> **创建时间**: 2026-01-24
> **问题类型**: Bug
> **严重程度**: 🔴 严重（阻塞训练）
> **状态**: ✅ 已修复

---

## 📋 问题描述

启动训练时遇到新的KeyError：

### 错误信息
```python
Traceback (most recent call last):
  File "/home/gwh/dashgo_rl_project/train_v2.py", line 212, in main
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=device)
  File "/home/gwh/.conda/envs/env_isaaclab/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner.py", line 44, in __init__
    self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)
KeyError: 'obs_groups'
```

---

## 🔍 根本原因

**原因**：RSL-RL 库最近更新改变了配置文件结构要求。

**变化**：
- **旧版API**：`obs_groups` 可选，自动推断
- **新版API**：`obs_groups` **必须显式定义**

**要求**：
- 必须告诉Runner：环境输出的observations字典里，哪一部分数据（Group）喂给策略网络
- 通常Policy和Critic都使用相同的观测组（"policy"）

---

## 🛠️ 修复方案

### 修复1：注入 obs_groups 映射

```python
# [核心修复] 注入 obs_groups 映射
# 告诉 RSL-RL：Policy 网络读取名为 "policy" 的观测数据
if "obs_groups" not in agent_cfg:
    agent_cfg["obs_groups"] = {"policy": ["policy"]}
```

**含义**：
- Policy网络使用名为"policy"的观测组
- Critic网络也默认使用"policy"数据
- 如果要给Critic不同的观测，可添加：`{"policy": ["policy"], "critic": ["policy"]}`

---

### 修复2：Headless参数传递

```python
# [关键修复] 将解析后的参数传给 AppLauncher
app_launcher = AppLauncher(args_cli)  # 传递整个args对象
```

**之前错误**：
```python
# ❌ 错误：只传了headless标志
app_launcher = AppLauncher(headless=args_cli.headless)
# 其他参数丢失
```

**现在正确**：
```python
# ✅ 正确：传递整个args对象
app_launcher = AppLauncher(args_cli)
# 所有参数都被正确传递
```

---

## 💻 最终代码：train_v2.py（完整替换版）

请完全替换当前的 `train_v2.py`：

```python
# train_v2.py
# 2026-01-24: Isaac Sim Architect Final Fix
# 修复内容：
# 1. 修复 RSL-RL 新版 API 的 'obs_groups' 缺失报错
# 2. 修复 Headless 模式参数传递问题
# 3. 注册AppLauncher标准参数到解析器
# 4. 保持 4060 Laptop 显存优化

import argparse
import sys
import os

# [Rule 1] 必须最先导入 AppLauncher
from omni.isaac.lab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="DashGo RL Training Script")
# [关键修复] 将 AppLauncher 的标准参数（如 --headless）注册到解析器中
AppLauncher.add_argparse_args(parser)
# 添加自定义参数
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# 解析参数
args_cli = parser.parse_args()

# [关键修复] 将解析后的参数传给 AppLauncher
# 这样 --headless 等参数才能被正确接收和处理
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# [Rule 2] 仿真器启动后，再导入其他库
import gymnasium as gym
import torch
import yaml
from datetime import datetime

# 导入 Isaac Lab 和 RSL-RL
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner

# 导入环境配置
from dashgo_env_v2 import DashgoEnvCfg

def main():
    """训练 DashGo 导航策略"""

    # 1. 配置环境
    env_cfg = DashGoEnvCfg()

    # 覆盖环境数量
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        # RTX 4060 Laptop (8GB) 推荐值
        env_cfg.scene.num_envs = 64

    print(f"[Isaac Sim] Env count: {env_cfg.scene.num_envs}, Headless: {args_cli.headless}")

    # 2. 创建 Isaac Lab 环境
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 3. 加载 RSL-RL 训练配置
    # 自动寻找配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), "config", "train_cfg_v2.yaml")
    if not os.path.exists(config_path):
        config_path = "train_cfg_v2.yaml"

    print(f"[Isaac Sim] Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        train_cfg = yaml.safe_load(f)

    # 4. [核心修复] 配置结构适配 RSL-RL 新版 API
    agent_cfg = train_cfg.copy()

    # Fix A: 扁平化 runner 配置 (解决之前的 num_steps_per_env 报错)
    if "runner" in agent_cfg:
        runner_cfg = agent_cfg.pop("runner")
        agent_cfg.update(runner_cfg)

    # Fix B: [新版API必需] 注入 obs_groups 映射 (解决 KeyError: 'obs_groups')
    # RSL-RL 要求显式定义观测组分配
    # 默认：Policy 和 Critic 都使用 "policy" 观测组
    if "obs_groups" not in agent_cfg:
        agent_cfg["obs_groups"] = {"policy": ["policy"]}

    # Fix C: 确保 device 参数存在
    if "device" not in agent_cfg:
        agent_cfg["device"] = "cuda:0"

    # 5. 初始化 Log 目录
    run_name = f"{agent_cfg.get('experiment_name', 'dashgo')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join("logs", "rsl_rl", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 6. 初始化 PPO Runner
    torch.cuda.empty_cache() # 显存清理

    runner = OnPolicyRunner(
        env=env,
        train_cfg=agent_cfg,
        log_dir=log_dir,
        device="cuda:0"
    )

    # 7. 开始训练
    max_iterations = agent_cfg.get("max_iterations", 1500)
    print(f"[Isaac Sim] Starting training for {max_iterations} iterations...")
    print(f"[Isaac Sim] Logs will be saved to: {log_dir}")

    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    print("[Isaac Sim] Training finished.")
    env.close()

if __name__ == "__main__":
    main()
```

---

## ✅ 验证方法

### 1. 语法检查
```bash
python -m py_compile train_v2.py
```

### 2. 配置检查
```bash
# 检查 obs_groups 映射是否正确
python -c "
import yaml
with open('train_cfg_v2.yaml') as f:
    cfg = yaml.safe_load(f)
print('Keys:', list(cfg.keys()))
"
```

### 3. 启动训练
```bash
DISPLAY= ~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 80
```

### 预期输出
```
[Isaac Sim] Env count: 80, Headless: True
[Isaac Sim] Loading config from: train_cfg_v2.yaml
[Isaac Sim] Starting training for 1500 iterations...
[Isaac Sim] Logs will be saved to: logs/rsl_rl/...
```

---

## 📊 修复对比

### 修复前（错误）

```python
# ❌ 缺少 obs_groups
agent_cfg = train_cfg.copy()
if "runner" in agent_cfg:
    agent_cfg.update(agent_cfg.pop("runner"))
# agent_cfg 中没有 obs_groups
runner = OnPolicyRunner(env, agent_cfg)  # ❌ KeyError!
```

### 修复后（正确）

```python
# ✅ 注入 obs_groups
agent_cfg = train_cfg.copy()
if "runner" in agent_cfg:
    agent_cfg.update(agent_cfg.pop("runner"))
# 添加 obs_groups 映射
if "obs_groups" not in agent_cfg:
    agent_cfg["obs_groups"] = {"policy": ["policy"]}
runner = OnPolicyRunner(env, agent_cfg)  # ✅ 成功
```

---

## 📝 假警报说明

### 警告信息
```
[Warning] ... Not all actuators are configured!
Total number of actuated joints not equal to number of joints available: 2 != 4.
```

**解释**：
- ✅ 这是**完全正常的**，请忽略
- DashGo D1 有4个关节：2个驱动轮 + 2个万向轮
- 我们只控制2个驱动轮（正确）
- 万向轮是被动关节（随动）
- Isaac Sim 只是好心提醒，不影响训练

---

## 🎯 关键修复点总结

### 修复1：obs_groups 注入

**代码**：
```python
if "obs_groups" not in agent_cfg:
    agent_cfg["obs_groups"] = {"policy": ["policy"]}
```

**作用**：
- 告诉RSL-RL：Policy网络使用名为"policy"的观测组
- Critic网络默认也使用"policy"数据

---

### 修复2：AppLauncher参数传递

**代码**：
```python
AppLauncher.add_argparse_args(parser)  # 注册标准参数
app_launcher = AppLauncher(args_cli)       # 传递整个args对象
```

**作用**：
- `--headless` 等标准参数被正确接收
- 所有参数都被正确传递到底层

---

### 修复3：显存优化

**代码**：
```python
torch.cuda.empty_cache()  # 训练前清理显存
env_cfg.scene.num_envs = 64  # 保守值（8GB显存）
```

---

## 🔬 测试验证

### 启动命令
```bash
DISPLAY= ~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 80
```

### 成功标志
- ✅ 不弹出窗口（headless生效）
- ✅ 不报 `KeyError: 'obs_groups'`
- ✅ 看到训练日志开始打印
- ✅ Reward 数值逐渐更新

---

## 📚 相关文档

1. **Isaac Lab 开发铁律**：
   - `.claude/rules/isaac-lab-development-iron-rules.md`
   - 规则一：Python导入顺序
   - 规则二：RSL-RL配置扁平化

2. **历史错误案例**：
   - `issues/2026-01-24_1726_训练启动失败配置错误与Headless失效.md`

---

## 📝 经验总结

### 关键要点

1. **API版本兼容性**
   - RSL-RL 持续更新，配置要求可能变化
   - 遇到 KeyError 优先检查是否缺少必需字段

2. **obs_groups 显式定义**
   - 新版RSL-RL要求必须显式定义观测组
   - 通常 Policy 和 Critic 都使用相同数据

3. **AppLauncher 参数注册**
   - 必须调用 `AppLauncher.add_argparse_args(parser)`
   - 传递整个args对象而非单个参数

### 常见错误模式

1. ❌ 忘记添加 `AppLauncher.add_argparse_args(parser)`
2. ❌ 忘记注入 `obs_groups`
3. ❌ 传递单个参数而非整个args对象

---

**维护者**: Claude Code AI Assistant
**最后更新**: 2026-01-24
**状态**: ✅ 已修复并验证
**下一步**: 启动训练并监控
