# TensorDict 运行时未解包 - IndexError

> **发现时间**: 2026-01-27 16:20:00
> **严重程度**: 🔴严重（训练启动后立即崩溃）
> **状态**: ✅已解决
> **相关文件**: `geo_nav_policy.py`

---

## 问题描述

在训练启动后，第一步采集数据时发生索引错误。初始化成功，但运行时崩溃。

### 完整错误信息

```python
[INFO] 开始训练: dashgo_v5_auto
[INFO] 环境数量: 16
[INFO] 单次采集步数: 24
[INFO] 最大迭代次数: 8000
------------------------------------------------------------
Traceback (most recent call last):
  File "/home/gwh/dashgo_rl_project/train_v2.py", line 353, in main
    runner.learn(num_learning_iterations=agent_cfg.get("max_iterations", 1500), init_at_random_ep_len=True)
  File "/home/gwh/.conda/envs/isaaclab/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner.py", line 103, in learn
    actions = self.alg.act(obs)
  File "/home/gwh/.conda/envs/isaaclab/lib/python3.10/site-packages/rsl_rl/algorithms/ppo.py", line 133, in act
    self.transition.actions = self.policy.act(obs).detach()
  File "/home/gwh/dashgo_rl_project/geo_nav_policy.py", line 146, in act
    self.update_distribution(observations)
  File "/home/gwh/dashgo_rl_project/geo_nav_policy.py", line 179, in update_distribution
    mean = self.forward_actor(observations)
  File "/home/gwh/dashgo_rl_project/geo_nav_policy.py", line 116, in forward_actor
    lidar = obs[:, :self.num_lidar].unsqueeze(1)  # [N, 1, 216]
  File "/home/gwh/.conda/envs/isaaclab/lib/python3.10/site-packages/tensordict/base.py", line 597,    in __getitem__
    return self._index_tensordict(index)
  File "/home/gwh/.conda/envs/isaaclab/lib/python3.10/site-packages/tensordict/_td.py", line 1628, in _index_tensordict
    batch_size = _getitem_batch_size(batch_size, index)
  File "/home/gwh/.conda/envs/isaaclab/lib/python3.10/site-packages/tensordict/utils.py", line 1883, in _getitem_batch_size
    batch = batch_size[count]
IndexError: tuple index out of range
```

### 错误位置

**文件**：`geo_nav_policy.py`
**方法**：`forward_actor()`
**行号**：第 116 行（修复前）
**代码**：`lidar = obs[:, :self.num_lidar].unsqueeze(1)`

---

## 根本原因

### 问题本质：TensorDict 解包不完整

**架构师诊断**：

这是一个典型的 **TensorDict 解包** 错误。

**问题分析**：

1. **初始化成功**：
   - `__init__` 中的 TensorDict 处理是正确的
   - 维度推断成功：`Actor总维数: 246`
   - 网络构建完成

2. **运行时失败**：
   - `act(observations)` 和 `evaluate(critic_observations)` 收到的仍是 TensorDict
   - `forward_actor()` 中直接对 TensorDict 进行切片 `[:, :216]`
   - TensorDict 只有 batch 维度 [16]，没有特征维度
   - 索引 `[:, :216]` 超出 TensorDict 的维度范围

**为什么会这样**：

```python
# ❌ 错误理解
obs = Tensor({
    'policy': Tensor[16, 246]
})
# 以为可以直接切片：obs[:, :216] → 失败！

# ✅ 正确理解
obs = TensorDict({
    'policy': Tensor[16, 246]  # 特征维度在这里！
}, batch_size=[16])
# 需要先解包：obs['policy'] → Tensor[16, 246]
# 然后切片：obs['policy'][:, :216] → 成功 ✅
```

**TensorDict 结构**：
- **外层**：TensorDict（类似字典），只有 batch 维度
- **内层**：Tensor（真正的数据），有 batch + feature 维度
- **错误**：对外层直接切片，忽略了内层结构

---

## 解决方案

### 核心思路：运行时统一解包

**架构师方案**：添加 `_extract_tensor()` 辅助方法，在运行时统一解包 TensorDict

### 实施步骤

#### 1. 记住 policy_key（初始化时）

```python
def __init__(self, obs, obs_groups, num_actions, ...):
    # ✅ 不仅提取维度，还记住使用的 key
    if hasattr(obs, "get"):
        self.policy_key = "policy" if "policy" in obs.keys() else list(obs.keys())[0]
        policy_tensor = obs[self.policy_key]
    else:
        self.policy_key = None
        policy_tensor = obs

    self.num_actor_obs = policy_tensor.shape[1]
```

**好处**：
- 避免运行时重复查找 key
- 提高 `_extract_tensor()` 的效率
- 只在初始化时检查一次

#### 2. 添加辅助方法（统一解包）

```python
def _extract_tensor(self, obs):
    """
    [Helper] 从 TensorDict 中提取 Tensor

    输入:
        obs: 可能是 TensorDict 或 Tensor

    输出:
        Tensor: 纯粹的张量，shape=[N, feature_dim]
    """
    if self.policy_key and hasattr(obs, "get"):
        # TensorDict: {'policy': Tensor[N, 246], ...}
        return obs[self.policy_key]
    # 已经是 Tensor 或其他情况
    return obs
```

**特点**：
- ✅ 兼容 TensorDict 和 Tensor
- ✅ 使用缓存的 `self.policy_key`
- ✅ 简洁、高效、可复用

#### 3. 修改前向传播（先解包再切片）

```python
def forward_actor(self, obs):
    # [Fix] 运行时解包 TensorDict -> Tensor
    x = self._extract_tensor(obs)  # ✅ 先解包

    # 现在 x 是纯粹的 Tensor，可以安全切片
    # x shape: [Batch, 246]

    # 1. 数据切片
    lidar = x[:, :self.num_lidar].unsqueeze(1)  # [Batch, 1, 216] ✅
    state = x[:, self.num_lidar:]               # [Batch, 30]

    # 2. 后续处理（不变）
    geo_feat = self.geo_encoder(lidar)
    ...
    return mu
```

#### 4. 修改 Critic 评估（也需要解包）

```python
def evaluate(self, critic_observations, **kwargs):
    # [Fix] Critic 也需要解包 TensorDict
    x = self._extract_tensor(critic_observations)  # ✅ 解包
    return self.critic(x)  # ✅ 评估
```

---

## 验证方法

### 1. 语法检查
```bash
python -m py_compile geo_nav_policy.py
```

### 2. 训练启动测试
```bash
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --enable_cameras --num_envs 16
```

**预期结果**：
- ✅ 初始化成功：`[GeoNavPolicy] 维度推断完成`
- ✅ 训练开始：`[INFO] 开始训练: dashgo_v5_auto`
- ✅ 第一步采集成功：不再出现 `IndexError`
- ✅ 进度条显示：`Iteration 1/8000`

---

## 经验教训

### 1. TensorDict 的两层结构

**教训**：理解 TensorDict 的嵌套结构至关重要

**结构示意**：
```
TensorDict (外层容器)
├── batch_size: [16]
├── keys: ['policy', 'critic', ...]
└── data (内层数据)
    ├── 'policy': Tensor[16, 246]  ← 真正的数据在这里
    ├── 'critic': Tensor[16, 246]
    └── ...
```

**操作规则**：
- ❌ 不能直接切片外层：`obs[:, :216]`
- ✅ 必须先访问内层：`obs['policy'][:, :216]`

### 2. 初始化 vs 运行时

**教训**：TensorDict 在两个阶段都需要处理

**修复历史回顾**：
1. **初始化阶段**（commit `dc556e4`）：
   - `__init__` 接收 TensorDict
   - 提取维度：`policy_tensor.shape[1]`
   - ✅ 已修复

2. **运行时阶段**（本次）：
   - `act()` 和 `evaluate()` 接收 TensorDict
   - 前向传播需要解包
   - ✅ 本次修复

**完整修复**：
```python
# 初始化
def __init__(self, obs, obs_groups, ...):
    self.policy_key = obs.keys()[0]  # 记住 key
    self.num_actor_obs = obs[self.policy_key].shape[1]  # 推断维度

# 运行时
def forward_actor(self, obs):
    x = obs[self.policy_key]  # 解包（使用记住的 key）
    lidar = x[:, :216]  # 切片
```

### 3. 辅助方法的价值

**教训**：使用辅助方法统一处理重复逻辑

**对比**：

**❌ 分散处理（容易遗漏）**：
```python
def forward_actor(self, obs):
    if hasattr(obs, 'get'):
        obs = obs['policy']
    lidar = obs[:, :216]

def evaluate(self, obs):
    # ❌ 忘记解包了！
    return self.critic(obs)
```

**✅ 统一处理（不容易遗漏）**：
```python
def _extract_tensor(self, obs):
    if self.policy_key and hasattr(obs, "get"):
        return obs[self.policy_key]
    return obs

def forward_actor(self, obs):
    x = self._extract_tensor(obs)  # ✅ 统一解包
    lidar = x[:, :216]

def evaluate(self, obs):
    x = self._extract_tensor(obs)  # ✅ 自动解包
    return self.critic(x)
```

**优势**：
- 集中管理解包逻辑
- 避免重复代码
- 减少遗漏风险

---

## 相关提交

- **Commit**: `445518e` - fix: 添加运行时TensorDict解包 - 修复训练时IndexError
- **文件修改**:
  - `geo_nav_policy.py`: 添加 `_extract_tensor()` 方法
  - `forward_actor()`: 调用 `_extract_tensor()` 解包
  - `evaluate()`: 调用 `_extract_tensor()` 解包

---

## 相关问题

### 前置问题
1. `2026-01-27_1545_actorcritic参数传递冲突_TypeError.md` - 关键字参数修复
2. `2026-01-27_1600_rslrl版本冲突_ActorCritic参数缺失.md` - 断开继承修复
3. `2026-01-27_1610_tensorsdict类型不匹配_维度推断失败.md` - TensorDict 接口适配

### 修复历史
1. **修复1**：关键字参数（commit `6e11be3`）- 部分解决
2. **修复2**：断开继承（commit `63be9d5`）- 版本兼容性
3. **修复3**：TensorDict 接口适配（commit `dc556e4`）- 初始化阶段
4. **修复4**：TensorDict 运行时解包（commit `445518e`）- 运行时阶段 ✅

---

## 参考资料

### TensorDict 官方文档
- TensorDict: PyTorch 的数据结构，用于管理张量字典
- 访问方式：`td["key"]` 返回 Tensor
- batch_size: 外层容器的 batch 维度
- 特征维度：内层 Tensor 的第二维

### RSL-RL 数据流
```
环境 → TensorDict({ 'policy': Tensor[16, 246] })
      ↓
OnPolicyRunner.learn()
      ↓
PPO.act(obs) → obs 仍是 TensorDict
      ↓
GeoNavPolicy.act(observations) → observations 仍是 TensorDict
      ↓
forward_actor(observations) → 需要解包！
```

### 架构师诊断原文
> "这是一个典型的 **TensorDict 解包** 错误。
>
> **现象**：
> `IndexError: tuple index out of range` 发生在 `lidar = obs[:, :self.num_lidar]`。
>
> **原因**：
>
> * 你的 `obs` 是一个 **TensorDict** 对象（类似字典的容器），而不是一个纯粹的 **Tensor**（张量）。
> * 当你试图对 TensorDict 进行二维切片 `[:, :216]` 时，它以为你要切分 batch 维度，但 TensorDict 通常只有 batch 维度（[16]），没有"特征维度"（246）。特征维度藏在字典的 value 里。
> * 我们在 `__init__` 里修复了维度推断，但**忘记在 `forward_actor` 运行时也进行同样的解包操作**。"

---

**文档维护**：此问题已解决并归档
**最后更新**: 2026-01-27 16:20:00
**归档原因**: 完整适配 TensorDict（初始化 + 运行时），训练可以正常启动
