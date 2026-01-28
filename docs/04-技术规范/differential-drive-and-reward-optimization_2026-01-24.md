# 架构师建议：差速驱动与奖励函数优化

> **文档类型**: 参考建议
> **创建时间**: 2026-01-24
> **状态**: 待评估
> **优先级**: 中

---

## 📋 建议概述

本建议来自一位 Isaac Sim 架构师，针对 DashGo RL Navigation 项目提出了**差速驱动映射**和**奖励函数稳定性**问题的重构方案。

**⚠️ 重要提示**: 这些建议仅供参考，尚未经过项目验证。实施前需要：
1. 评估与现有代码的兼容性
2. 测试对训练稳定性的影响
3. 对比性能提升幅度

---

## 🎯 核心建议

### 1. 环境配置 (dashgo_env_cfg.py)

**问题诊断**:
- 早期 Isaac Gym 直接控制关节速度 → Agent 难以学会差速运动学
- PhysX Lidar 在大规模并行训练中 GPU 显存占用高

**建议方案**:
- ✅ 使用 `DifferentialDriveAction` 逻辑（或 Action Term 映射）
- ✅ 直接将 Agent 动作映射为左右轮速度
- ✅ **强制使用 `RayCaster` 替代 `PhysX Lidar`**
  - GPU 显存占用降低 40%
  - 提升 FPS

**历史问题**:
- ❌ 机器人原地高速旋转
- ❌ LiDAR 数据穿模

---

### 2. 奖励函数 (dashgo_rewards.py)

**架构师强烈警告**:
> 🚨 **严厉禁止使用 "Orientation Reward"（朝向奖励）**
>
> **原因**: 在无数次实验中，朝向奖励只会导致局部最优解——机器人原地打转以完美对准目标，却不移动。

**建议方案**:
- ✅ 采用 **Tracking Reward (速度追踪)** 模式
- ✅ 计算所需的 \( v_x \) 和 \( \omega \)
- ✅ 奖励机器人匹配这个速度

**对比**:
| 方案 | 优点 | 缺点 |
|------|------|------|
| 朝向奖励 | 实现简单 | ❌ 导致原地转圈（局部最优） |
| 速度追踪 | 鲁棒，避免局部最小 | 实现稍复杂 |

---

### 3. RSL-RL 训练配置 (dashgo_ppo_cfg.yaml)

**参数调整**:
```yaml
# 学习率调整（针对低速差速机器人）
learning_rate: 1e-3  # 从 1e-4 提高到 1e-3

# 新增参数
normalize_advantage: true  # Isaac Lab 4.5 保持收敛平滑的关键
```

**理由**:
- DashGo 低速差速机器人不需要像 ANYmal 四足机器人那样激进的学习率
- `normalize_advantage` 有助于收敛曲线平滑

---

## 🔧 标准化开发范式

架构师提出的三个原则：

### 1. 物理与控制分离
- ✅ 在 `env_cfg` 中通过 `JointVelocityAction` 配置
- ✅ 使用 `ActuatorCfg` 的 `damping` 参数模拟真实电机特性
- ❌ 不依赖纯理想物理

### 2. 奖励哲学
- ❌ 移除所有可能导致 Local Minima 的朝向奖励
- ✅ 转而使用更鲁棒的速度追踪（Velocity Tracking）

### 3. 传感器升级
- ❌ 弃用 `PhysX Lidar`
- ✅ 全面拥抱 `RayCaster`
- 📈 显著提升仿真 FPS

---

## 📚 参考资源

### 视频教程
**[Tutorial #6 (Part 1) – Differential Drive in Isaac Sim](https://www.youtube.com/watch?v=WqNkmz0BnzQ)**

**内容**:
- 在 Isaac Sim 中配置差速驱动机器人的基础
- 虽然 RSL-RL，但底层 Action Graph 和 Joint 配置原理一致

### Isaac Lab 文档
- `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/dashgo/`

---

## 🚀 实施建议（如果采用）

### 实施步骤
1. **备份当前代码**
   ```bash
   git checkout -b backup/current-implementation
   git push origin backup/current-implementation
   ```

2. **创建特性分支**
   ```bash
   git checkout -b feature/differential-drive-optimization
   ```

3. **分阶段实施**
   - 阶段1: 替换 PhysX Lidar → RayCaster
   - 阶段2: 修改奖励函数（移除朝向奖励）
   - 阶段3: 调整学习率参数

4. **对比测试**
   - 训练 1000 iterations
   - 对比 reward 曲线
   - 检查是否出现原地转圈

5. **性能基准**
   - 测量 GPU 显存占用
   - 测量训练 FPS
   - 验证是否达到"显存降低 40%" 的目标

---

## ⚖️ 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 速度追踪实现复杂 | 可能引入新bug | 参考官方示例，逐步测试 |
| 移除朝向奖励可能影响收敛 | 训练时间延长 | 保留朝向奖励分支，A/B测试 |
| RayCaster 与 headless 模式冲突 | 需要额外适配 | 检查 RayCaster 文档 |
| 学习率调整导致不稳定 | 训练崩溃 | 使用 W&B 监控梯度 |

---

## 📝 当前项目状态对比

### 当前实现 (dashgo_env_v2.py)
- ✅ 已使用 `UniDiffDriveAction`（自定义差速驱动）
- ⚠️ 使用 Camera 传感器（伪 LiDAR）
- ⚠️ 奖励函数包含对准奖励（`reward_facing_target`）
- ✅ 学习率 1e-4（保守值）

### 建议实现
- ✅ 使用标准 `DifferentialDriveAction`
- ✅ 使用 RayCaster 传感器
- ❌ 完全移除对准奖励
- ✅ 学习率 1e-3（激进值） + normalize_advantage

---

## 🔍 待验证问题

1. **RayCaster 与 headless 兼容性**
   - 当前 Camera 传感器在 headless 模式有问题
   - RayCaster 是否支持 headless？
   - 需要查询 Isaac Lab 文档

2. **速度追踪奖励的实现细节**
   - 具体公式是什么？
   - 如何计算目标 \( v_x \) 和 \( \omega \)？
   - 是否有官方示例可参考？

3. **朝向奖励的权重问题**
   - 当前朝向奖励权重已降至 0.1（commit history）
   - 是否仍导致原地转圈？
   - 需要实际训练数据验证

4. **学习率调整的必要性**
   - 当前 1e-4 已经很保守
   - 提高 10 倍到 1e-3 是否真的有收益？
   - 是否会导致训练不稳定？

---

## ✅ 行动清单

在采用这些建议之前：

- [ ] 查阅 Isaac Lab 4.5 RayCaster 官方文档
- [ ] 验证 RayCaster headless 兼容性
- [ ] 测试当前奖励函数是否仍导致原地转圈
- [ ] 对比学习率 1e-4 vs 1e-3 的训练稳定性
- [ ] 查找 RSL-RL 官方速度追踪奖励示例
- [ ] 评估 GPU 显存占用（基线测试）
- [ ] 测量当前训练 FPS（基线测试）

---

## 📌 备注

- 本建议未经过实际验证
- 任何修改都需要 A/B 对比测试
- 保留当前实现作为 fallback
- 文档会持续更新，记录所有架构建议

---

**维护者**: Claude Code AI Assistant
**最后更新**: 2026-01-24
**版本**: v1.0
