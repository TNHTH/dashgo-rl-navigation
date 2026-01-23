# DashGo RL Navigation Project

## 项目简介

基于深度强化学习的DashGo机器人局部导航项目，使用NVIDIA Isaac Lab和RSL-RL训练。

## 开发环境

- **仿真器**: NVIDIA Isaac Sim 4.5
- **操作系统**: Ubuntu 20.04 LTS
- **框架**: Isaac Lab 0.46.4 + RSL-RL
- **语言**: Python 3.10

## 项目结构

```
dashgo_rl_project/
├── train_v2.py              # 训练脚本
├── train_cfg_v2.yaml         # 训练配置
├── dashgo_env_v2.py          # 环境定义
├── dashgo_assets.py          # 机器人资产
├── dashgo_config.py          # ROS参数配置
├── play.py                  # 演示脚本
├── export_onnx.py           # 导出ONNX模型
├── run_headless_train.sh    # Headless训练启动脚本
├── check_config.py           # 配置验证脚本
├── issues/                  # 问题记录
├── docs/                    # 项目文档
└── dashgo/                  # 实物ROS包（参考）
```

## 快速开始

### 1. 训练新模型

```bash
# Headless模式（推荐，80个并行环境）
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 80

# 或使用专用脚本
./run_headless_train.sh --num_envs 80
```

### 2. 演示已有模型

```bash
# 自动加载最新模型
~/IsaacLab/isaaclab.sh -p play.py --num_envs 1

# 指定模型
~/IsaacLab/isaaclab.sh -p play.py --checkpoint logs/model_450.pt --num_envs 1
```

### 3. 导出ONNX模型

```bash
python export_onnx.py --checkpoint logs/model_450.pt
```

## 核心特性

### 严格的参数对齐
- 所有物理参数从ROS配置文件读取
- 确保仿真与实物一致
- 配置来源: `dashgo/EAI驱动/dashgo_bringup/config/`

### 官方文档优先
- 严格遵循Isaac Sim 4.5官方文档
- 所有API使用已验证在目标版本中存在
- 禁止使用未经验证的功能

### 问题记录系统
- 所有问题记录在 `issues/` 目录
- 包含详细的根本原因分析和解决方案
- 作为项目经验积累

## 最近修改

### 2024-01-24

1. **修复headless模式相机传感器问题**
   - headless模式下自动禁用相机传感器
   - 条件配置避免prim路径错误
   - 创建专用启动脚本

2. **优化参数管理**
   - 从ROS配置文件读取参数
   - 避免硬编码
   - 统一配置中心

3. **完善开发规范**
   - 创建问题记录系统
   - 添加项目级开发规则
   - 严格的版本锁定（Isaac Sim 4.5）

4. **修复训练配置**
   - 移除不支持的学习率调度参数
   - 降低朝向奖励权重防止原地转圈
   - 使用标准AppLauncher用法

## 提交历史

最近26个本地提交（待推送）：
- 问题修复：headless模式、参数配置、版本锁定
- 代码重构：ROS参数读取、AppLauncher标准用法
- 文档完善：问题记录、开发规范

详细提交历史请查看：[提交补丁](/tmp/dashgo_changes.patch)

## 注意事项

- ⚠️ headless模式下激光雷达观测被禁用（返回零值）
- ⚠️ 即使加--headless，Isaac Sim仍可能打开窗口（正常行为）
- ✅ 窗口打开不影响训练，可以最小化
- ✅ 训练正常使用GPU (cuda:0)

## 后续优化

- 使用RayCaster传感器替代相机（支持headless）
- 添加TensorBoard支持
- 优化奖励函数
- 完善单元测试

## 许可证

本项目仅供学习和研究使用。
