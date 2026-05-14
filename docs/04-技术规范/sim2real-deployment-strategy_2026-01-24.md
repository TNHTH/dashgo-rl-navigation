# 架构师建议：Sim2Real 实机部署方案

> **文档类型**: 参考建议
> **创建时间**: 2026-01-24
> **状态**: 待评估
> **建议来源**: Isaac Sim 架构师（Sim2Real专家）

---

## 📋 建议概述

本建议针对**仿真训练完美，实机落地就废**的经典问题，提供了一套基于**物理锚定+感知降维+动作整形**的Sim2Real方案。

**⚠️ 重要提示**: 这些建议仅供参考，尚未经过项目验证。

---

## 🎯 核心策略

### 1. 物理锚定 (Physics Anchoring)

**原理**：在 `DashGoSpecs` 中锁定轮径 \( r \) 和轮距 \( L \)，确保仿真运动学方程与真实机器人PID算法完全一致。

**建议代码**：
```python
from .dashgo_specs import DashGoSpecs

# 锁定参数
wheel_radius = DashGoSpecs.WHEEL_RADIUS  # 0.0632m
wheel_track = DashGoSpecs.WHEEL_TRACK    # 0.342m
```

**收益**：
- ✅ 仿真与实物运动学完全对齐
- ✅ 避免参数误差累积

---

### 2. 感知降维 (Observation Sparsification)

**原理**：参考 NeuPAN 论文，不输入 1000 个雷达点，而是输入 **16~32 个扇区的最小距离**。

**收益**：
- ✅ 过滤真实雷达噪点（玻璃反光等）
- ✅ 网络训练速度提升 10 倍
- ✅ 降低计算量

**当前实现**：
```python
# dashgo_env_v2.py: 使用180个点的Camera传感器
lidar_sensor = CameraCfg(
    height=1, width=180,  # 180个点
    data_types=["distance_to_image_plane"],
)
```

**建议修改**：
- 压缩到 16-32 个扇区
- 每个扇区取最小距离

---

### 3. 动作整形 (Action Shaping)

**问题**：神经网络输出 \( [-1, 1] \)，可能超出电机极限。

**建议**：在环境中硬截断（Hard Clip）
```python
# 伪代码：部署时的转换逻辑
cmd_v = model_output[0] * 0.3  # 严格限制最大线速度
cmd_w = model_output[1] * 1.0  # 严格限制最大角速度
```

**当前实现**：
```python
# dashgo_env_v2.py: 已有速度裁剪
max_lin_vel = MOTION_CONFIG["max_lin_vel"]  # 0.3 m/s
max_ang_vel = MOTION_CONFIG["max_ang_vel"]  # 1.0 rad/s
target_v = torch.clamp(actions[:, 0] * max_lin_vel, -max_lin_vel, max_lin_vel)
```

**对比**：✅ 当前实现已包含速度限制，符合建议

---

## 💻 建议的代码实现

### 场景配置 (DashgoSceneCfg)

**建议修改**：

```python
@configclass
class DashgoSceneCfg(InteractiveSceneCfg):
    # 地面：摩擦系数随机化
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.PhysicsMaterialCfg(
                friction=1.0,
                restitution=0.0
            )
        ),
    )

    # 机器人：加载 USD 资产
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Wait_for_my_asset_path/dashgo_d1.usd",
            activate_contact_sensors=True,
        ),
        actuators={
            "diff_drive": ArticulationCfg.ActuatorCfg(
                joint_names_expr=[".*_wheel_joint"],
                effort_limit=20.0,     # N·m
                velocity_limit=5.0,    # rad/s
                stiffness=0.0,
                damping=100.0,         # ⚠️ 增大阻尼（当前是5.0）
            ),
        },
    )

    # 传感器：RayCaster（替代Camera）
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/lidar",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            horizontal_fov_range=(-math.pi, math.pi),
            horizontal_res=1.0,  # 1度分辨率 = 360个点
        ),
        max_distance=12.0,
        debug_vis=False,
    )
```

**关键差异**：
- ⚠️ damping: 100.0（当前是5.0）
- ✅ 使用RayCaster（与第一次建议一致）

---

### 奖励函数 (DashgoRewardsCfg)

**建议修改**：

```python
@configclass
class DashgoRewardsCfg:
    # [核心] 前进进度奖励
    progress = RewTerm(func=rewards.progress_reward, weight=1.0)

    # [惩罚] 碰撞惩罚
    collision = RewTerm(func=rewards.collision_penalty, weight=-200.0)

    # [惩罚] 靠近障碍物（SDF）
    safe_distance = RewTerm(
        func=rewards.min_distance_penalty,
        weight=-0.5,
        params={"threshold": 0.25}
    )

    # [约束] 动作平滑度
    action_smoothness = RewTerm(func=rewards.action_rate_penalty, weight=-0.05)

    # [绝对禁止] orientation_tracking <-- 已删除
```

**当前实现**：
```python
# dashgo_env_v2.py
velodyne_style_reward = RewardTermCfg(...)  # 进度奖励
facing_goal = RewardTermCfg(
    func=reward_facing_target,
    weight=0.1,  # ⚠️ 朝向奖励（权重已降至0.1）
)
```

**关键差异**：
- ❌ 建议完全删除朝向奖励
- ⚠️ 当前仍有朝向奖励（权重0.1）

---

### 域随机化 (Domain Randomization)

**建议修改**：

```python
def __post_init__(self):
    # 质量随机化（模拟负载变化）
    self.events.randomize_mass = EventTerm(
        func=randomize_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_range": (-1.0, 3.0)  # ⚠️ 范围更大
        }
    )

    # 摩擦力随机化（地板/地毯/瓷砖）
    self.events.randomize_friction = EventTerm(
        func=randomize_friction,
        mode="interval",
        interval_s_range=(10.0, 20.0),
        params={
            "asset_cfg": SceneEntityCfg("ground"),
            "friction_range": (0.6, 1.2)  # ⚠️ 动态随机化
        }
    )
```

**当前实现**：
```python
# dashgo_env_v2.py
randomize_mass = EventTermCfg(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={"mass_distribution_params": (0.8, 1.2)},  # ±20%
)

physics_material = EventTermCfg(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "static_friction_range": (0.7, 1.3),  # 仅启动时
        "dynamic_friction_range": (0.7, 1.3),
    },
)
```

**关键差异**：
- ⚠️ 质量随机化范围更大：(-1.0, 3.0) vs (0.8, 1.2)
- ⚠️ 摩擦力动态随机化（interval模式） vs 仅启动时

---

### 传感器噪声

**建议**：在 `train_cfg_v2.yaml` 中添加

```yaml
noise:
  add_noise: True
  noise_level: 0.05  # 5cm 测量误差
```

**当前实现**：
```yaml
# train_cfg_v2.yaml
# ❌ 无噪声配置
```

**当前已实现**：
```python
# dashgo_env_v2.py
class PolicyCfg(ObservationGroupCfg):
    def __post_init__(self):
        self.enable_corruption = True  # ✅ 观测噪声已开启
```

**对比**：✅ 当前已开启观测噪声（与建议一致）

---

## 🛡️ 防呆检查（稳定性）

### 1. 解决原地转圈

**历史问题**：
```python
# ❌ 导致原地转圈的奖励
reward = -abs(target_angle - current_angle)
```

**建议方案**：
```python
# ✅ 使用进度奖励，只有移动才有分
progress = RewTerm(func=rewards.progress_reward, weight=1.0)
```

**当前实现**：
```python
# dashgo_env_v2.py
velodyne_style_reward = RewardTermCfg(
    func=reward_navigation_sota,  # 包含进度奖励
    weight=1.0,
)
facing_goal = RewardTermCfg(
    func=reward_facing_target,
    weight=0.1,  # ⚠️ 仍有朝向奖励（但权重很低）
)
```

**评估**：⚠️ 朝向奖励权重0.1已很低，可能不会再导致转圈，但仍需验证

---

### 2. 解决速度失控

**建议**：
```python
cmd_v = model_output[0] * 0.3  # 硬截断
cmd_w = model_output[1] * 1.0
```

**当前实现**：
```python
# dashgo_env_v2.py
max_lin_vel = 0.3
max_ang_vel = 1.0
target_v = torch.clamp(actions[:, 0] * max_lin_vel, -max_lin_vel, max_lin_vel)
```

**对比**：✅ 当前实现已包含速度裁剪，符合建议

---

### 3. 对齐传感器噪声

**建议**：
- 开启 `enable_corruption`
- 添加 5cm Lidar噪声

**当前实现**：
- ✅ `enable_corruption = True`
- ❌ 未明确 Lidar 噪声强度

---

## 📊 对比分析：当前 vs 建议

| 维度 | 当前实现 | 本次建议 | 第一次建议 | 可融合性 |
|------|---------|----------|-----------|----------|
| **物理参数锁定** | ✅ DashGoROSParams | ❌ 需新建DashGoSpecs | - | ⚠️ 重复 |
| **传感器类型** | Camera (180点) | RayCaster (360点) | RayCaster (180点) | ⚠️ 需验证 |
| **感知降维** | ❌ 无 | ✅ 16-32扇区 | ❌ 无 | ⚠️ 需实现 |
| **阻尼系数** | 5.0 / 10.0 | 100.0 | 10.0 | ⚠️ 差异大 |
| **朝向奖励** | 0.1权重 | ❌ 完全删除 | ❌ 完全删除 | ⚠️ 需测试 |
| **质量随机化** | ±20% | ±200% (-1.0~3.0) | - | ⚠️ 激进 |
| **摩擦力随机化** | 启动时 | 动态(interval) | - | ⚠️ 激进 |
| **观测噪声** | ✅ 已开启 | 5cm Lidar噪声 | ✅ 已开启 | ✅ 一致 |

---

## 🔍 关键发现

### ✅ 已符合建议的部分

1. **速度限制**：当前已有硬截断（0.3 m/s, 1.0 rad/s）
2. **观测噪声**：已开启 `enable_corruption`
3. **物理随机化**：已有质量和摩擦力随机化

### ⚠️ 需要评估的差异

1. **阻尼系数**：5.0 vs 100.0（20倍差异）
   - 影响：阻尼过大可能导致运动缓慢
   - 建议：保持当前值，或测试中间值（10.0, 20.0）

2. **质量随机化范围**：(0.8, 1.2) vs (-1.0, 3.0)
   - 影响：范围过大可能导致训练不稳定
   - 建议：保持当前值，或逐步扩大

3. **摩擦力随机化模式**：startup vs interval
   - 影响：interval模式更激进，但可能影响收敛
   - 建议：先保持startup，测试后考虑interval

4. **朝向奖励**：权重0.1 vs 完全删除
   - 影响：当前权重已很低，可能不再导致转圈
   - 建议：训练1000次迭代，观察是否转圈

### ❌ 新建议（未在其他建议中出现）

1. **感知降维**：16-32扇区压缩
   - 收益：计算量降低，训练速度提升
   - 风险：可能丢失信息
   - 建议：可选择性测试

2. **DashGoSpecs**：新建物理参数类
   - 问题：当前已有 `DashGoROSParams`
   - 建议：复用现有类，无需重复

---

## ⚠️ 风险评估

| 建议 | 风险 | 缓解措施 |
|------|------|----------|
| 阻尼100.0 | 运动缓慢 | 保持5.0，或测试10.0 |
| 质量±200% | 训练崩溃 | 保持±20%，或逐步扩大到±50% |
| 摩擦力interval | 收敛困难 | 保持startup，测试后考虑interval |
| 删除朝向奖励 | 收敛变慢 | 降低权重到0.01，观察效果 |
| 感知降维 | 信息丢失 | 对比180点vs32点性能 |

---

## 📋 行动建议

### 立即可做（低风险）

1. ✅ **保持当前实现**
   - 速度限制已正确
   - 观测噪声已开启
   - 物理随机化已完善

2. 📊 **训练基准测试**
   - 使用当前配置训练1000次迭代
   - 记录是否出现原地转圈
   - 记录reward曲线

### 需要实验验证（中风险）

3. ⚠️ **朝向奖励测试**
   - 如果基准测试无转圈 → 保持0.1权重
   - 如果转圈 → 降低到0.01或完全删除

4. ⚠️ **阻尼系数测试**
   - 保持当前5.0
   - 如果运动太滑 → 提高到10.0

### 高风险（谨慎）

5. ❌ **暂不采纳**
   - 质量随机化范围扩大到±200%
   - 摩擦力动态随机化（interval模式）
   - 阻尼系数提高到100.0

---

## 🔬 实验计划

### 阶段1：基准测试（当前配置）

```bash
# 使用当前配置训练
DISPLAY= ~/IsaacLab/isaaclab.sh -p apps/isaac/train_v2.py --headless --num_envs 80

# 监控指标：
# - 是否出现原地转圈
# - Reward曲线是否平滑
# - 最终成功率
```

### 阶段2：朝向奖励测试

```bash
# 创建测试分支
git checkout -b test/remove-facing-reward

# 修改朝向奖励权重为0
# weight=0.1 → weight=0.0

# 训练1000次迭代
# 对比性能
```

### 阶段3：感知降维测试（可选）

```bash
# 创建测试分支
git checkout -b test/perception-compression

# 修改观测处理
# 压缩180点到32个扇区

# 训练1000次迭代
# 对比训练速度和性能
```

---

## 📝 总结

### 核心共识

两次架构师建议的**共识点**：
1. ✅ **必须移除朝向奖励**（或权重极低）
2. ✅ **必须开启观测噪声**
3. ✅ **必须添加物理随机化**
4. ✅ **必须使用RayCaster**（替代Camera）

### 关键差异

| 第一次建议 | 本次建议 | 建议 |
|-----------|---------|------|
| 学习率1e-3 | 未提及 | 保持1e-4 |
| RayCaster 180点 | RayCaster 360点 | 先测试180点 |
| 摩擦力±30% | 摩擦力interval模式 | 保持startup |
| 朝向奖励删除 | 朝向奖励删除 | 降低权重到0.01 |

### 推荐方案

**保守方案**（当前实现 + 微调）：
1. ✅ 保持当前物理参数（阻尼5.0）
2. ✅ 保持当前随机化范围（质量±20%）
3. ⚠️ 朝向奖励权重：0.1 → 0.01（降低10倍）
4. 📊 训练1000次迭代，观察效果
5. 📊 如果无转圈，恢复到0.1

---

**维护者**: Claude Code AI Assistant
**最后更新**: 2026-01-24
**版本**: v1.0
**相关文档**:
- `differential-drive-and-reward-optimization_2026-01-24.md`
- `code-comparison-and-fusion-plan_2026-01-24.md`
