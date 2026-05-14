# 张量形状不匹配错误 - penalty_unsafe_speed 函数

> **创建时间**: 2026-01-27 16:15:00
> **严重程度**: 🟡 张量形状错误
> **状态**: ✅ 已解决
> **错误类型**: RuntimeError - Shape Mismatch
> **相关文件**: dashgo_env_v2.py

---

## 🚨 错误信息

### 完整错误堆栈

```
Traceback (most recent call last):
  File "/home/gwh/dashgo_rl_project/verify_collision.py", line 60, in main
    step_result = env.step(action)
  File "/home/gwh/IsaacLab/source/isaaclab/envs/manager_based_rl_env.py", line 209, in step
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
  File "/home/gwh/IsaacLab/source/isaaclab/managers/reward_manager.py", line 151, in compute
    self._reward_buf += value
RuntimeError: output with shape [1] doesn't match the broadcast shape [1, 1]
```

### 执行命令

```bash
~/IsaacLab/isaaclab.sh -p verify_collision.py --headless --enable_cameras
```

---

## 🔍 问题分析

### 根本原因

**架构师诊断**：
> 这是一个**张量形状不匹配（Shape Mismatch）**导致的崩溃。
>
> **错误核心**：`RuntimeError: output with shape [1] doesn't match the broadcast shape [1, 1]`
>
> **原因分析**：问题出在 `penalty_unsafe_speed` 函数中。
> 1. 你的代码：`min_dist = torch.min(all_dist, dim=1)[0]`
> 2. 数据现状：相机数据 `d_front` 是三维的 `[N, Height, Width]`（例如 `[1, 16, 90]`）
> 3. 计算后果：`torch.cat` 后变成了 `[1, 64, 90]`。你只对 `dim=1` 求了最小值，结果形状是 **`[1, 90]`**（保留了宽度维度），而不是我们要的一个标量 `[1]`
> 4. 崩溃时刻：当你把这个 `[1, 90]` 的惩罚矩阵加到 `[1]` 的总奖励缓冲区时，PyTorch 炸了

**详细解释**：

**原始代码（错误）**：
```python
# 1. 拼接 4 个相机 [N, H, W] each
all_dist = torch.cat([d_front, d_left, d_back, d_right], dim=1).squeeze(1)
# 结果：[N, 360]

# 2. 求最小值
min_dist = torch.min(all_dist, dim=1)[0]
# 结果：[N] ✅ 正确

# 问题：如果 squeeze(1) 没生效，或者数据本身就是 [N, H, W]
# 拼接后是 [N, 4*H*W]，dim=1 求最小值得到 [N, W]，而不是 [N]
```

**相机数据形状**：
- 原始：`[N, 1, 90]`（已 squeeze 过高度）
- 但实际可能是：`[N, 16, 90]`（未 squeeze）

**数据流**：
```
[N, H, W] cat → [N, 4*H, W]
                 ↓ dim=1, min → [N, W]  ← 错误！
```

---

## ✅ 解决方案

### 修改文件：dashgo_env_v2.py

**位置**：`penalty_unsafe_speed` 函数

**修改前**（错误）：
```python
# 拼接并 squeeze
all_dist = torch.cat([d_front, d_left, d_back, d_right], dim=1).squeeze(1)  # [N, 360]
min_dist = torch.min(all_dist, dim=1)[0]  # [N]
```

**修改后**（正确）：
```python
# 拼接并展平为一维
batch_size = d_front.shape[0]
all_pixels = torch.cat([d_front, d_left, d_back, d_right], dim=1).view(batch_size, -1)  # [N, 4*H*W]

# 过滤 inf 并求最小值
all_pixels = torch.nan_to_num(all_pixels, posinf=12.0)
min_dist = torch.min(all_pixels, dim=1)[0]  # [N]
```

**关键改动**：
1. 使用 `.view(batch_size, -1)` 明确展平为二维
2. 添加 `nan_to_num` 处理 `inf` 值
3. 确保 `min_dist` 形状是 `[N]` 而不是 `[N, W]`

---

## 📊 技术细节

### 张量形状变化

**相机数据**：
- `d_front`: `[N, H, W]` = `[1, 16, 90]`
- `d_left`, `d_back`, `d_right`: 同上

**拼接后**：
```python
torch.cat([d_front, d_left, d_back, d_right], dim=1)
# 结果：[N, 4*H, W] = [1, 64, 90]
```

**错误处理（旧代码）**：
```python
.squeeze(1)  # 假设能去掉 H 维度
# 实际：H 可能不是 1，所以 squeeze 失败
torch.min(..., dim=1)[0]  # 结果：[N, W] = [1, 90]
```

**正确处理（新代码）**：
```python
.view(batch_size, -1)  # 强制展平为二维
# 结果：[N, 4*H*W] = [1, 5760]
torch.min(..., dim=1)[0]  # 结果：[N] = [1]
```

### inf 处理

**为什么需要过滤 inf？**
- 深度相机可能返回 `inf`（无限远）
- `torch.min` 会选择 `inf` 作为最小值
- 导致 `min_dist = inf`，逻辑错误

**解决方案**：
```python
all_pixels = torch.nan_to_num(all_pixels, posinf=12.0)
# 将 inf 替换为 12.0（最大探测距离）
```

---

## 🎓 经验教训

### 1. 张量形状必须明确

**教训**：不能假设 `squeeze` 能去除正确的维度
- ❌ 依赖 `.squeeze(1)` 自动去除维度
- ✅ 使用 `.view(batch_size, -1)` 明确展平

**原则**：
- 始终知道你的张量形状
- 使用 `view` 或 `reshape` 明确转换
- 不要假设 squeeze 能成功

### 2. 处理边界值

**教训**：必须处理 `inf` 和 `nan`
- 深度相机可能返回 `inf`（未探测到物体）
- 数学运算（`min`, `max`）会被 `inf` 干扰
- 使用 `nan_to_num` 替换为合理的值

### 3. 调试技巧

**教训**：遇到 Shape Mismatch 时
1. 打印张量形状：`print(tensor.shape)`
2. 检查每一步的形状变化
3. 使用 `torch.autograd.set_detect_anomaly(True)` 检测异常

---

## 🔧 实施记录

### 修改文件
- `dashgo_env_v2.py`：修改 `penalty_unsafe_speed()` 函数

### 修改内容
1. 改用 `.view(batch_size, -1)` 明确展平
2. 添加 `nan_to_num` 处理 `inf` 值
3. 添加详细注释说明形状变化

### 相关提交
- commit: aa355c6 (2026-01-27)

---

## 🚀 验证方法

### 清理残留进程

```bash
# 杀死残留进程（防止崩溃锁死显存）
pkill -9 -f "verify_collision.py"
pkill -9 -f "kit"
```

### 再次运行验证

```bash
~/IsaacLab/isaaclab.sh -p verify_collision.py --headless --enable_cameras
```

**预期输出**（成功）：
```text
Step 40: 速度=0.28 m/s | 接触力=12.3421 N | Done=False
...
🛑 [检测到重置] 在 Step XX 触发！
--------------------------------------------------
🕵️‍♂️ 重置原因取证:
   > 碰撞 (object_collision): 1.0  ← 关键指标
   > 翻车 (base_height):      0.0
   > 超时 (time_out):         0.0
--------------------------------------------------
✅ 验证成功：系统检测到了碰撞并触发了重置！
```

**只要看到这一行，你就通过了所有的"Sim-to-Real 环境图灵测试"，可以立即开启大规模训练了！**

---

## 📚 相关文档

- **三层防御体系**：`issues/2026-01-27_避障策略优化-三层防御体系_2026-01-27.md`
- **ContactSensor 修复**：`issues/2026-01-27_ContactSensor-API属性名错误_2026-01-27.md`
- **实施记录**：`issues/2026-01-27_Geo-Distill-V2.2实施记录.md`

---

**维护者**: Claude Code AI System (Robot-Nav-Architect Agent)
**项目**: DashGo机器人导航（Sim2Real）
**开发基准**: Isaac Sim 4.5 + Ubuntu 20.04
**状态**: ✅ 已解决，待验证
