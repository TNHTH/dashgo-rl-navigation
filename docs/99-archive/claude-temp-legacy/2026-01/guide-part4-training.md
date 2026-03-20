# DashGo RL Navigation 项目完全复现指南 v5.0

> **创建时间**: 2026-01-28 22:35:00
> **第四部分**: 训练实战指南
> **预计时间**: 15-25分钟
> **依赖**: 第三部分（架构解析）已完成

---

## 4.1 训练前检查清单

### Isaac Lab开发铁律（5条）

在启动训练前，**必须**检查以下5条铁律。违反任何一条都会导致训练失败或系统崩溃。

#### 铁律1: AppLauncher导入顺序

**检查项**：
```bash
# 检查train_v2.py第18-25行
head -n 25 train_v2.py | grep -A 5 "import"

# 应该看到：
# 1. from omni.isaac.lab.app import AppLauncher  # 必须在前5行
# 2. app_launcher = AppLauncher(headless=args.headless)
# 3. simulation_app = app_launcher.app
# 4. 然后才能导入torch、gymnasium等
```

**❌ 错误示例**：
```python
import torch  # ❌ 太早了！
from omni.isaac.lab.envs import ManagerBasedRLEnv  # ❌ 太早了！
from omni.isaac.lab.app import AppLauncher  # ❌ 太晚了！
```

**✅ 正确示例**：
```python
from omni.isaac.lab.app import AppLauncher  # ✅ 必须最先
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
import torch  # ✅ 现在可以导入了
from omni.isaac.lab.envs import ManagerBasedRLEnv
```

**违反后果**：
- ❌ headless参数失效（窗口弹出）
- ❌ 训练无法启动
- ❌ Segfault（段错误）

---

#### 铁律2: RSL-RL配置扁平化

**检查项**：
```bash
# 检查train_v2.py第80-95行
grep -A 10 "agent_cfg" train_v2.py | grep "pop\|update"

# 应该看到：
# agent_cfg.pop("runner")
# agent_cfg.update(runner_cfg)
```

**验证配置文件**：
```bash
# 检查train_cfg_v2.yaml结构
cat train_cfg_v2.yaml | head -20

# ✅ 允许嵌套（为了可读性）
# runner:
#   num_steps_per_env: 24

# ❌ 但Python代码必须扁平化处理
```

**违反后果**：
- ❌ `KeyError: 'num_steps_per_env'`
- ❌ 训练器初始化失败

---

#### 铁律3: 显存管理（RTX 4060）

**检查项**：
```bash
# 检查num_envs配置
grep "num_envs" train_cfg_v2.yaml

# ✅ 推荐：≤ 128（RTX 4060安全值）
# ❌ 禁止：> 128（会OOM）
```

**显存监控脚本**：
```bash
# 另一个终端监控显存
watch -n 1 nvidia-smi

# 正常范围：
# GPU利用率：80-95%
# 显存占用：6-7GB（留1-2GB余量）
# 温度：< 80°C
```

**违反后果**：
- ❌ OOM（Out of Memory）
- ❌ 训练速度崩溃（1000 FPS → 0.1 FPS）
- ❌ 系统卡死

---

#### 铁律4: 物理参数对齐

**检查项**：
```bash
# 检查是否从ROS配置读取参数
grep -n "DashGoROSParams\|from_yaml" dashgo_config.py

# 应该看到：
# ros_params = DashGoROSParams.from_yaml("dashgo/EAI驱动/...")
# wheel_radius = ros_params.wheel_radius  # 0.0632
```

**参数验证**：
```bash
# 检查关键参数
python -c "
from dashgo_config import DashGoROSParams
params = DashGoROSParams.from_yaml()
print(f'轮径: {params.wheel_radius} m')  # 应为0.0632
print(f'轮距: {params.wheel_track} m')   # 应为0.3420
print(f'最大速度: {params.max_lin_vel} m/s')  # 应为0.3
"
```

**违反后果**：
- ❌ Sim2Real完全失败（仿真策略无法部署到实物）
- ❌ 里程计误差累积（1%轮径误差=10cm定位误差）
- ❌ 机器人运动轨迹偏移

---

#### 铁律5: USD坐标系验证

**检查项**：
```bash
# 在Isaac Sim GUI中验证USD文件
cd $ISAACSIM_PATH
./isaac-sim.sh

# 在GUI中：
# 1. File → Import → 选择dashgo_d1.urdf
# 2. 检查机器人是否自然平放在地面
# 3. 轮子是否与地面接触（无悬空、无陷入）
# 4. 机器人是否侧躺或倒着
```

**违反后果**：
- ❌ Episode瞬间结束（检测到"碰撞"）
- ❌ 训练无法收敛（机器人一直"翻车"）
- ❌ Reward持续为负

---

### 完整检查清单

```bash
# 创建检查脚本
cat > pre_training_check.sh << 'EOF'
#!/bin/bash

echo "=== 训练前检查清单 ==="

# 1. AppLauncher导入顺序
echo "1. 检查AppLauncher导入顺序..."
if head -n 25 train_v2.py | grep -q "from omni.isaac.lab.app import AppLauncher"; then
    echo "✅ AppLauncher导入正确"
else
    echo "❌ AppLauncher导入顺序错误"
fi

# 2. RSL-RL配置扁平化
echo "2. 检查配置扁平化代码..."
if grep -q "agent_cfg.pop" train_v2.py; then
    echo "✅ 配置扁平化代码存在"
else
    echo "❌ 缺少配置扁平化代码"
fi

# 3. 显存管理
echo "3. 检查num_envs配置..."
num_envs=$(grep "num_envs" train_cfg_v2.yaml | awk '{print $2}')
if [ $num_envs -le 128 ]; then
    echo "✅ num_envs=$num_envs (≤128，安全)"
else
    echo "⚠️ num_envs=$num_envs (>128，可能OOM)"
fi

# 4. 物理参数对齐
echo "4. 检查ROS参数对齐..."
if [ -f "dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml" ]; then
    echo "✅ ROS配置文件存在"
else
    echo "❌ ROS配置文件缺失"
fi

# 5. USD坐标系（手动检查）
echo "5. USD坐标系验证（需手动在GUI中检查）"
echo "   请在Isaac Sim GUI中打开dashgo_d1.urdf验证"

echo "=== 检查完成 ==="
EOF

chmod +x pre_training_check.sh
./pre_training_check.sh
```

**全部通过** → 可以启动训练
**有❌项目** → 修复后再训练

---

## 4.2 训练配置详解

### train_cfg_v2.yaml完整解析

```yaml
# === 算法配置（PPO超参数）===
algorithm:
  # 学习率（控制参数更新幅度）
  learning_rate: 3.0e-4      # ✅ 标准值（RSL-RL推荐1e-4到1e-3）
                             # ⚠️ 太高→训练不稳定，太低→收敛慢

  # 熵系数（控制探索）
  entropy_coef: 0.01         # ✅ 标准值（0.005-0.02）
                             # ⚠️ 太高→随机探索，太低→早熟收敛

  # PPO裁剪参数
  clip_param: 0.2            # ✅ PPO标准值（不要修改）
                             # 防止策略更新过大

  # GAE参数
  gamma: 0.99                # 折扣因子（未来奖励权重）
  lambd: 0.95                # GAE平滑因子

# === 策略网络配置 ===
policy:
  # 网络类名（必须注入到RSL-RL）
  class_name: "GeoNavPolicy"  # 自定义轻量网络

  # Actor网络（策略网络）
  actor_hidden_dims: [128, 64]   # 轻量级（适合Jetson Nano）
                                 # ⚠️ 太大→推理慢，太小→性能差

  # Critic网络（价值网络）
  critic_hidden_dims: [512, 256, 128]  # 强力裁判
                                       # Critic可以比Actor大

# === 训练器配置 ===
runner:
  # 每个环境的步数
  num_steps_per_env: 24       # ✅ 标准值（16-32）
                             # ⚠️ 太大→显存占用高，太小→样本效率低

  # 训练轮数
  max_iterations: 8000        # 充分收敛（建议≥5000）

  # 保存频率
  save_interval: 500          # 每500轮保存一次

  # 经验归一化（自动归一化观测）
  empirical_normalization: True  # ✅ 推荐开启（加速收敛）

# === 环境配置 ===
env:
  # 并行环境数量
  num_envs: 64               # ✅ RTX 4060安全值（≤128）
                             # ⚠️ >128可能OOM

  # Episode时长
  episode_length_s: 20.0     # 每个Episode20秒

# === 课程学习配置 ===
curriculum:
  # 目标范围扩展
  target_expansion:
    min_range: 0.5           # 初始目标距离（米）
    max_range: 3.0           # 最终目标距离（米）
    end_step: 8640000        # 75%训练时完成（自动计算）
```

### 关键参数含义

| 参数 | 含义 | 标准值 | 调整建议 |
|------|------|--------|----------|
| **learning_rate** | 学习率 | 3e-4 | 训练爆炸→降到1e-4，收敛慢→提高到5e-4 |
| **entropy_coef** | 熵系数 | 0.01 | 探索不足→提高到0.02，太随机→降到0.005 |
| **actor_hidden_dims** | Actor网络层数 | [128,64] | 推理慢→减小[64,32]，性能差→增大[256,128] |
| **num_steps_per_env** | 每轮步数 | 24 | 显存不足→降到16，显存充足→提高到32 |
| **num_envs** | 并行环境数 | 64 | OOM→降到32，显存有余→提高到128 |

---

## 4.3 启动训练（headless模式）

### 基础训练命令

```bash
# 激活环境
conda activate env_isaaclab

# 进入项目目录
cd ~/dashgo_rl_project

# 启动训练（headless模式）
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 64
```

### 完整训练命令（所有参数）

```bash
~/IsaacLab/isaaclab.sh \
  -p train_v2.py \                # 训练脚本
  --headless \                    # 无GUI模式
  --num_envs 64 \                 # 并行环境数量（RTX 4060推荐64）
  --experiment dashgo_v5_auto \   # 实验名称
  --device cuda:0                 # 训练设备（cuda:0或cpu）
```

### 预期输出

```
[INFO] 启动Isaac Sim...
[INFO] 创建环境: num_envs=64
[GeoNavPolicy v3.1] 检测到 TensorDict，使用键: 'policy'
[GeoNavPolicy v3.1] 最终架构确认:
  - 输入维度: 246 (LiDAR=216, 状态=30)
  - 动作维度: 2 (线速度、角速度)
  - 梯度爆炸防护: LayerNorm + Input Clamp + Orthogonal Init
[INFO] 开始训练...
Learning iteration 1/8000
  - Mean Reward: -12.5
  - Mean Episode Length: 45.2
  - Policy Noise: 0.82
  - Value Loss: 5.23
...
```

---

## 4.4 监控训练过程

### TensorBoard使用

#### 启动TensorBoard

```bash
# 另一个终端
conda activate env_isaaclab
cd ~/dashgo_rl_project

# 启动TensorBoard
tensorboard --logdir logs/dashgo_v5_auto --port 6006

# 浏览器访问
# http://localhost:6006
```

#### 关键指标解读

**1. Mean Reward（平均奖励）**
- **理想曲线**: 持续上升，最终稳定在正值
- **问题诊断**:
  - ❌ 持续下降 → 训练爆炸，降低learning_rate
  - ❌ 剧烈波动 → entropy_coef太高或太低
  - ❌ 长期不增长 → 奖励函数设计问题

**2. Mean Episode Length（平均Episode长度）**
- **理想曲线**: 逐渐增长（机器人学会走更远）
- **正常范围**: 50-200步
- **问题诊断**:
  - ❌ 持续很短（<50步）→ 机器人一直碰撞
  - ❌ 持续很长（>500步）→ 机器人原地转圈

**3. Policy Noise（策略噪声）**
- **含义**: 策略输出的变化幅度
- **正常范围**: 0.1-1.0
- **问题诊断**:
  - ❌ 持续增长（>10）→ 训练爆炸，立即停止！
  - ❌ 持续很低（<0.01）→ 策略早熟，提高entropy_coef

**4. Value Loss（价值损失）**
- **含义**: Critic网络的预测误差
- **正常范围**: 1-10
- **问题诊断**:
  - ❌ 持续增长（>100）→ 价值网络发散

### 终端实时监控

```bash
# 监控训练日志
tail -f logs/dashgo_v5_auto/log.txt

# 监控GPU状态
watch -n 1 nvidia-smi

# 监控训练进度
watch -n 10 'ls -lth logs/dashgo_v5_auto/models/ | head'
```

---

## 4.5 Checkpoint管理

### Checkpoint保存机制

RSL-RL自动保存checkpoint：
```bash
# 每500轮保存一次（train_cfg_v2.yaml配置）
logs/dashgo_v5_auto/
├── models/
│   ├── model_500.pt      # 第500轮
│   ├── model_1000.pt     # 第1000轮
│   └── ...
└── logs/
    └── events.out.tfevents.*  # TensorBoard日志
```

### 恢复训练

```bash
# 从checkpoint恢复训练
~/IsaacLab/isaaclab.sh \
  -p train_v2.py \
  --headless \
  --num_envs 64 \
  --resume \
  --checkpoint logs/dashgo_v5_auto/models/model_450.pt
```

### 模型选择

**如何选择最佳模型？**
```bash
# 方法1: 查看TensorBoard，选择Mean Reward最高的checkpoint
# 方法2: 查看训练日志
grep "Mean Reward" logs/dashgo_v5_auto/log.txt | tail -20

# 示例输出：
# Iteration 4500: Mean Reward = 85.2
# Iteration 5000: Mean Reward = 92.7  ← 最佳
# Iteration 5500: Mean Reward = 89.1
```

---

## 4.6 常见训练问题解决

### 问题1: 训练爆炸（Policy Noise > 10）

**错误现象**：
```
Policy Noise: 26.82 → 17.30 → 15.67 (持续增长)
Value Loss: 152.3 → 892.1 → 1205.7 (爆炸)
```

**解决方案**：
```yaml
# 修改train_cfg_v2.yaml
algorithm:
  learning_rate: 1.5e-4   # 从3e-4降到1.5e-4
  entropy_coef: 0.005     # 从0.01降到0.005

rewards:
  shaping_distance:
    weight: 0.5           # 从2.0降到0.5（引导奖励太高）
  collision:
    weight: -50.0         # 从-20.0加重到-50.0
```

---

### 问题2: 训练不收敛（Mean Reward持续为负）

**错误现象**：
```
Mean Reward: -15.2 → -18.5 → -20.1 → -22.3 (持续下降)
```

**可能原因**：
1. **奖励函数设计问题**
2. **学习率太低**
3. **环境配置错误**

**解决方案**：
```yaml
# 1. 检查奖励权重
rewards:
  reach_goal:
    weight: 2000.0  # 确保到达目标奖励足够大

# 2. 提高学习率
algorithm:
  learning_rate: 5.0e-4  # 从3e-4提高到5e-4

# 3. 检查环境配置
env:
  episode_length_s: 20.0  # 确保Episode时长足够
```

---

### 问题3: 显存溢出（OOM）

**错误现象**：
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**解决方案**：
```yaml
# 降低并行环境数量
env:
  num_envs: 32  # 从64降到32

# 或降低批量大小
runner:
  num_steps_per_env: 16  # 从24降到16
```

---

### 问题4: 机器人原地转圈

**错误现象**：
```
Mean Episode Length: 500+ (机器人一直转圈，不撞墙也不到达目标)
```

**可能原因**：
1. **朝向奖励存在**（会导致转圈）
2. **目标向量计算错误**

**解决方案**：
```python
# 检查奖励函数，确保没有朝向奖励
# ❌ 错误：有orientation奖励
rewards["orientation"] = 0.5

# ✅ 正确：移除朝向奖励，改用势能差
rewards["progress_to_goal"] = 1.0
```

---

### 问题5: 训练速度慢（<100 FPS）

**错误现象**：
```
FPS: 45.2  # 正常应该>100
```

**可能原因**：
1. **CPU瓶颈**（物理仿真）
2. **GPU利用率低**
3. **num_envs太小**

**解决方案**：
```bash
# 1. 增加并行环境
env:
  num_envs: 80  # 从64提高到80

# 2. 检查GPU利用率
nvidia-smi dmon -s u -c 1
# 应该看到GPU利用率>80%

# 3. 如果CPU瓶颈，考虑降低物理精度
sim:
  dt: 0.02  # 从0.1降到0.02（更精确但更慢）
```

---

### 问题6: 机器人一直撞墙

**错误现象**：
```
Mean Episode Length: 5-10 (机器人刚启动就碰撞)
Collision Rate: 95%
```

**可能原因**：
1. **碰撞惩罚太轻**
2. **传感器配置错误**
3. **动作空间太大**

**解决方案**：
```yaml
# 1. 加重碰撞惩罚
rewards:
  collision:
    weight: -100.0  # 从-50.0加重到-100.0

# 2. 降低速度限制
robot:
  max_lin_vel: 0.2  # 从0.3降到0.2 m/s

# 3. 检查传感器数据
# 在play.py中可视化传感器输出
```

---

### 问题7: Episode异常结束

**错误现象**：
```
Episode terminated unexpectedly: Missing observation key
```

**可能原因**：
1. **传感器数据缺失**
2. **观测空间配置错误**

**解决方案**：
```python
# 检查传感器配置
# dashgo_env_v2.py中确保传感器名称正确
env.scene["camera_front"]  # 确保与USD中一致

# 检查观测空间
print(env.observation_space)
# 应该包含所有观测键
```

---

### 问题8: 模型无法加载（Checkpoint损坏）

**错误现象**：
```
FileNotFoundError: checkpoint file not found
或
RuntimeError: Error loading model
```

**解决方案**：
```bash
# 1. 检查checkpoint文件
ls -lh logs/dashgo_v5_auto/models/
# 应该看到model_*.pt文件

# 2. 重新训练（从零开始）
rm -rf logs/dashgo_v5_auto
~/IsaacLab/isaaclab.sh -p train_v2.py --headless

# 3. 或从早期checkpoint恢复
~/IsaacLab/isaaclab.sh -p train_v2.py --headless \
  --resume --checkpoint logs/dashgo_v5_auto/models/model_500.pt
```

---

### 问题9: 学习曲线震荡

**错误现象**：
```
Mean Reward: 50 → 80 → 30 → 90 → 40 → 85 (剧烈波动)
```

**可能原因**：
1. **学习率太高**
2. **熵系数不稳定**
3. **批量大小太小**

**解决方案**：
```yaml
# 1. 降低学习率
algorithm:
  learning_rate: 1e-4  # 从3e-4降到1e-4

# 2. 稳定熵系数
algorithm:
  entropy_coef: 0.005  # 固定值，不要用衰减

# 3. 增加批量大小
runner:
  num_steps_per_env: 32  # 从24增加到32
```

---

### 问题10: TensorBoard无数据

**错误现象**：
```
浏览器打开TensorBoard显示"No dashboards found"
```

**解决方案**：
```bash
# 1. 检查日志目录
ls -l logs/dashgo_v5_auto/logs/
# 应该看到events.out.tfevents.*文件

# 2. 检查TensorBoard启动路径
tensorboard --logdir logs/dashgo_v5_auto  # 确保路径正确

# 3. 清除浏览器缓存
# Chrome → Ctrl+Shift+Delete → 清除缓存
```

---

## 4.7 训练完成判断

### 何时停止训练？

**标准1: Mean Reward稳定**
```
最近500轮Mean Reward波动<10%
例如：85 ± 5（稳定在80-90之间）
```

**标准2: Episode达标**
```
成功率 > 80%
Mean Episode Length > 100步
```

**标准3: 达到最大迭代**
```
Learning iteration 8000/8000
```

### 训练成功标志

```
✅ Mean Reward > 50（正值）
✅ Mean Episode Length > 100
✅ Success Rate > 80%
✅ Policy Noise < 1.0（稳定）
✅ Value Loss < 10（收敛）
```

---

## 4.8 下一步

**恭喜！** 你已经学会了：

✅ 训练前检查清单（5条铁律）
✅ 训练配置详解（每个参数的含义）
✅ 启动训练（headless模式）
✅ 监控训练过程（TensorBoard）
✅ 常见训练问题解决（10个+）

**下一部分**：Sim2Real部署完整流程

我们将一起：
- 导出TorchScript模型
- 准备ROS环境
- 部署到Jetson Nano
- 实物测试与调试

**预计时间**：20-30分钟

---

**第四部分完成** | 总进度: 57% (4/7)
