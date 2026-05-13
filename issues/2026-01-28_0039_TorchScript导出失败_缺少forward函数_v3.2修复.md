# TorchScript导出失败 - 缺少forward函数

**问题ID**: 2026-01-28_0039
**严重程度**: 🚨 致命（阻塞部署）
**状态**: ✅ 已解决
**修复版本**: v3.2

---

## 问题描述

### 错误信息

```
NotImplementedError: Module [GeoNavPolicy] is missing the required "forward" function
```

### 发生场景

在执行模型导出脚本 `export_torchscript.py` 时：

```bash
~/IsaacLab/isaaclab.sh -p export_torchscript.py
```

### 完整错误堆栈

```
[INFO] 创建GeoNavPolicy v3.1网络...
[INFO] 加载权重: logs/model_7999.pt
✅ 权重加载成功

================================================================================
[EXPORT] 正在分析模型输入需求...
================================================================================
  • 网络类型: GeoNavPolicy v3.1
  • 期望输入Shape: [1, 246]
  • 期望输入Dtype: torch.float32
  • 设备: cuda:0

[INFO] 正在导出为TorchScript...
❌ 导出失败: Module [GeoNavPolicy] is missing the required "forward" function
Traceback (most recent call last):
  File "/home/gwh/dashgo_rl_project/export_torchscript.py", line 116, in main
    traced_model = torch.jit.trace(policy, dummy_input)
  File "/home/gwh/.conda/envs/env_isaaclab/lib/python3.10/site-packages/torch/jit/_trace.py", line 1002, in trace_module,
    return trace_module(
  ...
  File "/home/gwh/.conda/envs/env_isaaclab/lib/python3.10/site-packages/torch/nn/modules/module.py", line 387, in _forward_unimplemented
    raise NotImplementedError
```

---

## 根本原因分析

### 问题根源

**`torch.jit.trace`** 和 **ROS推理** 默认调用的是PyTorch模型的标准入口函数 —— **`forward()`**。

但是，`GeoNavPolicy` v3.1 只实现了RSL-RL特定的接口：
- `forward_actor()` - Actor前向传播
- `act_inference()` - 推理时的动作输出
- `evaluate()` - Critic前向传播

**缺少标准的 `forward()` 方法**，导致TorchScript无法导出。

### 设计冲突

| 框架 | 默认调用方法 | 用途 |
|------|--------------|------|
| **PyTorch** | `forward()` | 标准前向传播、TorchScript导出 |
| **RSL-RL** | `forward_actor()` | PPO训练的Actor前向传播 |

`GeoNavPolicy` v3.1 只实现了RSL-RL接口，忽略了PyTorch标准接口。

---

## 解决方案

### 架构师建议（已实施）

在 `geo_nav_policy.py` 中添加标准的 `forward()` 函数：

#### 1. 更新版本号

```python
# v3.1 → v3.2
# geo_nav_policy.py v3.2 - 梯度爆炸修复版 + TorchScript导出支持
```

#### 2. 添加标准forward()方法

在 `forward_actor()` 方法之后添加：

```python
# ======================================================================
# [v3.2 新增] 标准forward()函数 - 支持TorchScript导出
# ======================================================================
def forward(self, obs):
    """
    标准推理入口（用于TorchScript导出和实机部署）

    torch.jit.trace和ROS推理默认调用forward()方法

    Args:
        obs: 输入观测 Tensor [Batch, 246]

    Returns:
        mu: 动作均值 Tensor [Batch, 2]
    """
    # 兼容TensorDict输入
    x = self._extract_tensor(obs)

    # [v3.1] 输入截断：防止 Inf/NaN 进入网络
    x = torch.clamp(x, min=-10.0, max=10.0)

    # 数据切片
    lidar = x[:, :self.num_lidar].unsqueeze(1)  # [Batch, 1, 216]
    state = x[:, self.num_lidar:]               # [Batch, 30]

    # 视觉编码
    geo_feat = self.geo_encoder(lidar)

    # 特征融合
    fused = torch.cat([geo_feat, state], dim=1)
    h = self.fusion_layer(fused)

    # 推理
    h = self.memory_layer(h)

    # 输出
    mu = self.actor_head(h)
    return mu
```

#### 3. 保持接口兼容

```python
# 原有的RSL-RL接口保持不变
def forward_actor(self, obs):
    """RSL-RL训练接口"""
    # 调用forward()复用逻辑
    return self.forward(obs)

def act_inference(self, observations):
    """RSL-RL推理接口"""
    return self.forward(observations)
```

---

## 实施结果

### 修复后的输出

```bash
[GeoNavPolicy v3.2] 最终架构确认:
  - 输入维度: 246 (LiDAR=216)
  - 动作维度: 2
  - 梯度爆炸防护: LayerNorm + Input Clamp + Orthogonal Init
  - TorchScript导出: ✅ 支持标准forward()函数

✅ 模型已导出至: catkin_ws/src/dashgo_rl/models/policy_torchscript.pt
   模型大小: 5.23 MB
```

### Git提交

```
Commit: 86df1e8
Message: fix: 添加标准forward()函数 - 支持TorchScript导出和ROS推理

Changes:
- 添加标准forward()方法 (调用forward_actor逻辑)
- 兼容torch.jit.trace导出
- 兼容ROS推理（默认调用forward()）
- 版本号: v3.1 → v3.2
```

---

## 经验教训

### DR-001: PyTorch模型必须实现forward()

**规则**：
- 所有PyTorch模块（`nn.Module`）必须实现`forward()`方法
- 即使只用于训练，也要考虑未来导出和部署的需求

**正确实践**：
```python
class MyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 网络定义

    # ✅ 标准：必须实现
    def forward(self, obs):
        """PyTorch标准接口"""
        return self.network(obs)

    # ✅ 可选：添加别名
    def forward_actor(self, obs):
        """RSL-RL特定接口"""
        return self.forward(obs)
```

**错误实践**：
```python
class MyPolicy(nn.Module):
    def __init__(self):
        super().__init__()

    # ❌ 错误：只实现了特定框架接口
    def forward_actor(self, obs):
        """只有RSL-RL接口"""
        return self.network(obs)
```

### DR-002: 导出前必须验证接口兼容性

**检查清单**：
- [ ] 是否实现了`forward()`方法？
- [ ] `forward()`的输入输出维度是否正确？
- [ ] 是否兼容`torch.jit.trace()`？
- [ ] 是否兼容TorchScript？

**验证方法**：
```python
# 在导出脚本中测试
try:
    dummy_input = torch.randn(1, 246)
    output = policy(dummy_input)
    print(f"✅ forward()测试通过: {output.shape}")
except Exception as e:
    print(f"❌ forward()测试失败: {e}")
```

---

## 相关问题

### 相关issue

1. **[2026-01-27_1730] 梯度爆炸导致NaN错误**
   - 问题：缺少LayerNorm导致梯度爆炸
   - 解决：添加LayerNorm到所有层

2. **[2026-01-27_1930] 架构师建议 - 维度不匹配问题**
   - 问题：助手方案77维 vs 架构师246维
   - 解决：使用架构师完整方案

3. **[2026-01-28_0039] AGENTS.md文件丢失**
   - 问题：commit 1d0e2b9误删除
   - 解决：从git历史恢复

---

## 验证步骤

### 如何验证修复成功

```bash
# 1. 重新运行导出脚本
~/IsaacLab/isaaclab.sh -p export_torchscript.py

# 2. 检查输出
# 应该看到:
✅ 模型已导出至: catkin_ws/src/dashgo_rl/models/policy_torchscript.pt
   模型大小: 5.23 MB

# 3. 验证文件
ls -lh catkin_ws/src/dashgo_rl/models/policy_torchscript.pt
```

### 测试导出的模型

```python
import torch

# 加载导出的模型
model = torch.jit.load("catkin_ws/src/dashgo_rl/models/policy_torchscript.pt")

# 测试推理
dummy_input = torch.randn(1, 246)
output = model(dummy_input)

print(f"✅ 模型推理成功")
print(f"   输入shape: {dummy_input.shape}")
print(f"   输出shape: {output.shape}")
```

---

## 总结

### 问题

- **TorchScript导出失败**：缺少`forward()`函数
- **阻塞场景**：模型导出 → ROS部署 → 实物运行

### 解决方案

- ✅ 添加标准`forward()`方法
- ✅ 复用`forward_actor()`逻辑
- ✅ 保持所有RSL-RL接口兼容
- ✅ 版本号更新：v3.1 → v3.2

### 影响

- **代码变更**：47行新增，3行删除
- **向后兼容**：完全兼容，不影响训练
- **下一步**：可以继续导出和部署流程

---

**记录者**: TNHTH
**日期**: 2026-01-28 00:39
**状态**: ✅ 已解决并验证
