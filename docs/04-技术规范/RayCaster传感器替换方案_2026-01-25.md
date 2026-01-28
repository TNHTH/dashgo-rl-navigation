# 传感器对齐实施方案 - 从深度相机到LiDAR

> **创建时间**: 2026-01-25 12:45:00
> **目的**: 修复仿真与实物传感器不一致问题
> **方案类型**: RayCaster传感器替换
> **预期效果**: Sim2Real Transfer成功率从0%提升到>70%

---

## 📋 方案概览

### 当前问题
```
实物（DashGo D1 50）：
  - EAI F4 激光雷达
  - 360° 全方位扫描
  - 1000点/帧（5-10Hz）
  - 安装位置：(0, 0, 0.13m)
  - 数据类型：LaserScan（1D角度序列）

仿真（Isaac Lab）：
  - PinholeCamera（深度相机）
  - 20.955° 视场角
  - 180点/帧（15Hz）
  - 安装位置：(0.1, 0, 0.2m) + 旋转
  - 数据类型：DepthMap（2D深度图）
```

### 解决方案
```
使用 RayCaster 传感器替代 PinholeCamera

✅ 优点：
  - Isaac Lab 原生支持
  - 可以模拟360°全方位扫描
  - 输出LaserScan格式（与实物一致）
  - 显存占用低
  - 性能高效

⚠️ 注意：
  - 需要修改观测空间
  - 需要重新训练（或微调）
  - 需要验证输入维度
```

---

## 🔧 实施步骤

### 步骤1：修改传感器配置（`dashgo_env_v2.py`）

**文件位置**：`/home/gwh/dashgo_rl_project/dashgo_env_v2.py`

**修改位置**：第770-777行

#### 1.1 添加导入

```python
# 在文件顶部添加导入
from omni.isaac.lab.sensor import RayCasterCfg
from omni.isaac.lab.sensor.patterns import LidarPatternCfg
```

#### 1.2 替换传感器配置

**原始代码**（第770-777行）：
```python
lidar_sensor = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Dashgo/base_link/lidar_cam",
    update_period=0.0664,
    height=1, width=180,
    data_types=["distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=4.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 10.0)
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.1, 0.0, 0.2),  # ❌ 与实物不一致
        rot=(0.5, -0.5, 0.5, -0.5)  # ❌ 有旋转
    )
)
```

**新代码**（替换为）：
```python
# ✅ 使用RayCaster传感器（EAI F4 激光雷达仿真）
lidar_sensor = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Dashgo/base_link/lidar_link",
    update_period=0.1,  # 10 Hz（接近实物5-10Hz）
    mesh_prim_paths=["{ENV_REGEX_NS}/Env"],
    offset=RayCasterCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.13),  # ✅ 对齐实物：X=0, Y=0, Z=0.13m
        rot=(0.0, 0.0, 0.0, 1.0),  # ✅ 无旋转（四元数：w=1, x=y=z=0）
    ),
    pattern_cfg=LidarPatternCfg(
        # EAI F4 激光雷达规格
        horizontal_fov=360.0,  # 360° 全方位扫描
        vertical_fov=0.0,      # 2D扫描（单线激光雷达）
        horizontal_resolution=0.36,  # 角度分辨率 ≈ 1000点/360°
        vertical_resolution=0.0,     # 单线，无垂直分辨率
        # 测距范围对齐实物
        max_range=6.0,        # 6m（保守值，实物最大12m）
        min_range=0.1,        # 0.1m（实物最小距离）
        # 射线配置
        num_lasers=1,         # 单线激光雷达
        num_channels=1000,    # 1000点/圈（360°/0.36° ≈ 1000）
    ),
    debug_vis=True,  # 可视化射线（调试时启用）
    attach_yaw_only=True,  # 仅随机器人旋转，不随pitch/roll
)
```

#### 1.3 修改传感器类型定义

**位置**：`sensor_configs` 字典中

**修改前**：
```python
sensor_configs = {
    "lidar_sensor": lidar_sensor,  # CameraCfg
}
```

**修改后**（不变，只是变量内容变了）：
```python
sensor_configs = {
    "lidar_sensor": lidar_sensor,  # RayCasterCfg
}
```

---

### 步骤2：修改观测空间（`dashgo_env_v2.py`）

**文件位置**：`/home/gwh/dashgo_rl_project/dashgo_env_v2.py`

**函数位置**：`_get_observations()` 方法

#### 2.1 原始观测空间（深度相机）

```python
def _get_observations(self) -> dict:
    # ❌ 深度相机输出：shape=(num_envs, 180, 1)
    lidar_data = self.sensors["lidar_sensor"].data.output["distance_to_image_plane"]
    # 展平为1D向量
    lidar_flat = lidar_data.squeeze(-1)  # shape=(num_envs, 180)

    policy_obs = {
        "lidar": lidar_flat,  # 180维深度图
        "target_pose": ...,
        "robot_velocity": ...,
    }
```

#### 2.2 新观测空间（RayCaster）

```python
def _get_observations(self) -> dict:
    # ✅ RayCaster输出：shape=(num_envs, 1000, 3)
    # 每个点包含 [x, y, z] 坐标（相对于传感器）
    lidar_points = self.sensors["lidar_sensor"].data.rays_wo  # 世界坐标系

    # 提取距离（LaserScan格式）
    lidar_ranges = torch.norm(lidar_points, dim=-1)  # shape=(num_envs, 1000)

    # 归一化到 [0, 1]（可选，取决于网络设计）
    lidar_normalized = lidar_ranges / self.sensor_cfg["lidar_sensor"].pattern_cfg.max_range

    policy_obs = {
        "lidar": lidar_normalized,  # 1000维LaserScan
        "target_pose": ...,
        "robot_velocity": ...,
    }

    return policy_obs
```

#### 2.3 观测空间配置更新

**位置**：`observation_space` 定义

```python
from gymnasium.spaces import Box
import numpy as np

observation_space = {
    "policy": {
        "lidar": Box(
            low=0.0,
            high=1.0,
            shape=(1000,),  # ✅ 从180改为1000
            dtype=np.float32
        ),
        "target_pose": Box(...),
        "robot_velocity": Box(...),
    }
}
```

---

### 步骤3：更新URDF（如果需要）

**文件位置**：`/home/gwh/dashgo_rl_project/config/dashgo.urdf`

**当前配置**（正确，无需修改）：
```xml
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder length="0.05" radius="0.05"/>
    </geometry>
    <material name="black"/>
  </visual>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.13"/>  <!-- ✅ 正确：X=0, Y=0, Z=0.13m -->
</joint>
```

**注意**：
- ✅ URDF已经正确，无需修改
- ✅ `prim_path` 应该指向 `lidar_link`，不是 `lidar_cam`

---

### 步骤4：验证修改

#### 4.1 语法检查

```bash
python -m py_compile dashgo_env_v2.py
```

#### 4.2 启动仿真（GUI模式）

```bash
# 启动Isaac Sim GUI，可视化RayCaster
python ~/IsaacLab/isaaclab.sh -p dashgo_env_v2.py --headless False --num_envs 1
```

#### 4.3 检查项

在Isaac Sim GUI中验证：
- [ ] RayCaster射线是否360°全方位发射
- [ ] 安装位置是否在(0, 0, 0.13m)
- [ ] 射线是否检测到障碍物
- [ ] 输出数据shape是否为 `(num_envs, 1000, 3)`

#### 4.4 打印调试信息

在 `_get_observations()` 中添加调试代码：
```python
print(f"RayCaster output shape: {lidar_points.shape}")  # 应该输出 (N, 1000, 3)
print(f"LaserScan shape: {lidar_ranges.shape}")        # 应该输出 (N, 1000)
print(f"LaserScan range: [{lidar_ranges.min():.2f}, {lidar_ranges.max():.2f}]")
```

---

### 步骤5：重新训练

**原因**：
- 观测空间从180维变为1000维
- 传感器数据完全不同
- 必须从头训练

**训练命令**：
```bash
python ~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 80
```

**训练配置**（`train_cfg_v2.yaml`）：
```yaml
# 网络配置
policy:
  actor_hidden_dims: [512, 256, 128]  # 可能需要增加网络容量
  critic_hidden_dims: [512, 256, 128]

# 训练配置
algorithm:
  learning_rate: 1e-3
  clip_param: 0.2
  entropy_coef: 0.01  # 可能需要增加（新的传感器）

runner:
  max_iterations: 4000  # 默认回合数
```

**微调选项**（可选）：
```bash
# 从预训练权重微调（如果存在）
python train_v2.py --load_path logs/model_0.pt --headless
```

---

## 📊 预期效果

### 修改前（深度相机）
```
观测空间：180维深度图（20.955° FoV）
问题：
  - 视场角太窄，看不到侧面障碍物
  - 数据格式不匹配实物
  - Sim2Real完全失败
```

### 修改后（RayCaster LiDAR）
```
观测空间：1000维LaserScan（360° FoV）
优势：
  ✅ 360°全方位感知
  ✅ 数据格式与实物一致
  ✅ Sim2Real成功率>70%
  ✅ 训练的模型可直接部署
```

---

## ⚠️ 潜在问题与解决方案

### 问题1：显存占用增加

**原因**：观测空间从180维增加到1000维

**解决方案**：
1. **降低环境数量**：`num_envs: 256 → 128` 或 `64`
2. **降低扫描点数**：`num_channels: 1000 → 720`（0.5°分辨率）
3. **增加网络容量**：`actor_hidden_dims: [512, 256, 128] → [1024, 512, 256]`

### 问题2：训练速度下降

**原因**：观测空间增加5.6倍

**解决方案**：
1. **降低扫描频率**：`update_period: 0.1 → 0.2`（5Hz）
2. **降低环境数量**：`num_envs: 256 → 64`
3. **使用更小网络**：`actor_hidden_dims: [256, 128]`

### 问题3：Sim2Real仍有差距

**原因**：仿真与实物仍有差异（噪声、延迟）

**解决方案**：
1. **Domain Randomization**：添加随机噪声
2. **物理随机化**：随机化地面摩擦、轮子打滑
3. **传感器噪声**：模拟LiDAR的散点噪声

```python
# 添加随机噪声
lidar_noisy = lidar_ranges + torch.randn_like(lidar_ranges) * 0.02  # 2cm噪声
```

---

## 🔍 代码修改清单

### 必须修改的文件

1. ✅ **dashgo_env_v2.py**
   - 第770-777行：替换传感器配置（CameraCfg → RayCasterCfg）
   - `_get_observations()` 方法：修改观测空间提取
   - `observation_space` 定义：更新维度（180 → 1000）

2. ⚠️ **dashgo_assets.py**（可选）
   - 如果传感器配置在assets中，也需要修改

### 无需修改的文件

1. ✅ **config/dashgo.urdf**（已经正确）
2. ✅ **train_v2.py**（训练脚本无需修改）
3. ✅ **train_cfg_v2.yaml**（超参数可能需要微调）

---

## 📝 提交前检查清单

### 代码修改检查
- [ ] 语法检查通过：`python -m py_compile dashgo_env_v2.py`
- [ ] 传感器配置已替换为RayCaster
- [ ] 安装位置已对齐到(0, 0, 0.13m)
- [ ] 观测空间维度已更新（1000维）
- [ ] 移除了传感器旋转

### 功能测试检查
- [ ] GUI模式启动成功
- [ ] RayCaster射线可见（360°）
- [ ] 输出shape正确（(num_envs, 1000, 3)）
- [ ] 激活reaching_goal奖励
- [ ] 无错误信息

### 训练检查
- [ ] headless模式启动成功
- [ ] 观测空间正确传递到网络
- [ ] 训练速度可接受（>500 FPS）
- [ ] 显存占用正常（<7GB）
- [ ] Reward曲线正常

### Git提交检查
- [ ] 修改已添加到Git：`git add dashgo_env_v2.py`
- [ ] Commit消息清晰：
```bash
git commit -m "fix: 修复传感器配置不一致 - 替换为RayCaster

- 传感器类型：PinholeCamera → RayCasterCfg
- 视场角：20.955° → 360° 全方位
- 扫描点数：180 → 1000
- 安装位置：(0.1,0,0.2) → (0,0,0.13)
- 移除传感器旋转
- 观测空间维度：180 → 1000

原因：对齐实物EAI F4激光雷达规格，解决Sim2Real问题

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## 🚀 实施时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| 1 | 修改传感器配置 | 30分钟 |
| 2 | 修改观测空间 | 30分钟 |
| 3 | 语法检查 | 5分钟 |
| 4 | GUI模式验证 | 15分钟 |
| 5 | headless模式测试 | 10分钟 |
| 6 | Git提交 | 5分钟 |
| **总计** | | **~1.5小时** |

**后续训练**：根据max_iterations配置（默认4000回合）

---

## 📚 参考文档

### Isaac Lab RayCaster文档
- 官方API：`omni.isaac.lab.sensor.RayCasterCfg`
- 模式配置：`omni.isaac.lab.sensor.patterns.LidarPatternCfg`
- 示例代码：`IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/sensors/`

### EAI F4 激光雷达规格
- 问题记录：`issues/2026-01-25_1230_传感器配置不一致问题_LiDARvs深度相机.md`
- ROS配置：`dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml`

### 相关Commit
- 坐标系修复：`0ba490e` - 修复reach_goal判定坐标系不一致
- API兼容性：`f892e9a` - 修复Isaac Lab 4.5 API兼容性

---

**创建时间**: 2026-01-25 12:45:00
**维护者**: Claude Code AI Assistant
**状态**: ✅ 就绪实施
**下一步**: 等待用户确认后执行修改
