# DashGo RL Navigation 项目完全复现指南 v5.0

> **创建时间**: 2026-01-28 22:35:00
> **第一部分**: 快速开始
> **预计阅读时间**: 5分钟

---

## 1.1 项目概述

### 什么是DashGo RL Navigation？

DashGo RL Navigation是一个基于**深度强化学习**的机器人局部导航项目，旨在训练DashGo D1机器人实现自主避障和目标到达。

**核心特点**：
- ✅ **Sim2Real完整对齐**：仿真训练的策略可以直接部署到实物机器人
- ✅ **轻量级网络**：300K参数，适配Jetson Nano 4GB部署
- ✅ **4向相机LiDAR融合**：创新感知方案，低成本实现360°感知
- ✅ **自适应课程学习**：自动调整训练难度，提高收敛速度
- ✅ **梯度稳定**：v3.1网络架构，防止梯度爆炸

### 技术栈

| 组件 | 技术 | 版本（严格锁定） |
|------|------|------------------|
| 仿真环境 | NVIDIA Isaac Sim | 4.5 |
| 框架 | Isaac Lab | 0.46.4 |
| 算法 | RSL-RL (PPO) | v3.0.1 |
| 编程语言 | Python | 3.10 |
| 操作系统 | Ubuntu | 20.04 LTS |
| 部署硬件 | Jetson Nano | 4GB |
| ROS版本 | ROS Noetic | - |

**为什么要严格锁定版本？**
- Isaac Sim不同版本API可能不兼容
- Ubuntu 20.04是Jetson Nano官方支持版本
- 严格版本锁定确保文档中的所有命令都能正常运行

### 项目目标

**短期目标**：
- 训练一个能在仿真环境中导航的机器人
- 成功率 > 80%（到达目标且无碰撞）

**中期目标**：
- 导出TorchScript模型
- 部署到实物DashGo D1机器人

**长期目标**：
- 集成到ROS全局规划系统
- 实现长距离自主导航

### 项目定位

**重要**: 这个项目训练的是一个**局部路径规划器**（Local Planner），而非端到端导航器。

**局部规划器 vs 全局规划器**：

| 特性 | 局部规划器（本项目） | 全局规划器（ROS move_base） |
|------|---------------------|---------------------------|
| 作用范围 | 3-8米 | 全地图 |
| 输入 | LiDAR感知 + 目标方向 | 地图 + 全局路径 |
| 输出 | 立即执行的速度指令 | 全局路径点 |
| 响应速度 | 高频（10Hz+） | 低频（1Hz） |
| 职责 | 避障 + 短期导航 | 长距离寻路 |

**实际部署架构**：
```
用户指定目标点
    ↓
ROS move_base（全局规划器）生成全局路径
    ↓
DashGo RL Navigation（局部规划器）执行避障 + 局部导航
    ↓
实物机器人运动
```

---

## 1.2 系统要求检查清单

### 硬件要求

| 组件 | 最低配置 | 推荐配置 | 说明 |
|------|---------|---------|------|
| **GPU** | NVIDIA GTX 1660 (6GB) | NVIDIA RTX 4060 (8GB) | 用于仿真训练 |
| **CPU** | 4核心 | 8核心 | 物理仿真需要CPU计算 |
| **RAM** | 16GB | 32GB | Isaac Sim占用较大内存 |
| **存储** | 50GB可用空间 | 100GB SSD | Isaac Lab + 项目文件 |
| **实物部署** | Jetson Nano 4GB | Jetson Xavier NX | 可选，用于实物测试 |

**GPU兼容性**：
- ✅ 支持：RTX系列、GTX 16系列及以上
- ❌ 不支持：AMD GPU、Intel集成显卡
- 验证命令：`nvidia-smi`（应显示GPU信息）

### 软件环境

| 软件 | 版本 | 检查命令 |
|------|------|---------|
| **操作系统** | Ubuntu 20.04 LTS | `lsb_release -a` |
| **Python** | 3.10 | `python --version` |
| **CUDA** | 12.9 | `nvcc --version` |
| **Git** | 任意最新版 | `git --version` |
| **Conda** | Miniconda或Anaconda | `conda --version` |

**版本检查脚本**：
```bash
# 一键检查所有依赖
cat > check_requirements.sh << 'EOF'
#!/bin/bash
echo "=== 系统要求检查 ==="

echo -n "Ubuntu版本: "
lsb_release -d | cut -f2-

echo -n "Python版本: "
python --version 2>&1

echo -n "CUDA版本: "
nvcc --version 2>&1 | grep "release" | awk '{print $5}'

echo -n "Git版本: "
git --version

echo -n "Conda版本: "
conda --version 2>/dev/null || echo "未安装"

echo -n "GPU信息: "
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "未检测到GPU"

echo -n "显存大小: "
nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "N/A"
EOF

chmod +x check_requirements.sh
./check_requirements.sh
```

### 软件兼容性警告

⚠️ **以下配置不支持（会导致无法运行）**：
- ❌ Ubuntu 22.04或更高版本（Isaac Sim 4.5不支持）
- ❌ Python 3.11或更高版本（Isaac Lab依赖Python 3.10）
- ❌ 非NVIDIA GPU（Isaac Sim需要CUDA）
- ❌ Windows或macOS（仅支持Linux）

---

## 1.3 5分钟快速验证

### 目的
在正式安装前，先验证你的系统是否满足基本要求。

### 前置条件
假设你已经：
- ✅ 安装了Ubuntu 20.04 LTS
- ✅ 安装了NVIDIA GPU驱动
- ✅ 安装了Conda（Miniconda或Anaconda）

### 快速验证步骤

#### 步骤1: 创建Conda环境（1分钟）

```bash
# 创建专用环境
conda create -n test_isaaclab python=3.10 -y
conda activate test_isaaclab

# 验证Python版本
python --version
# 预期输出: Python 3.10.x
```

#### 步骤2: 安装PyTorch（2分钟）

```bash
# 安装PyTorch（CPU版本，仅用于测试）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 验证PyTorch
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
# 预期输出: PyTorch版本: 2.x.x
```

#### 步骤3: 测试CUDA（可选，1分钟）

```bash
# 如果有NVIDIA GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
# 预期输出: CUDA可用: True

# 如果显示False，检查GPU驱动
nvidia-smi
# 应显示GPU信息（型号、显存、驱动版本）
```

#### 步骤4: 克隆项目（1分钟）

```bash
# 克隆项目（如果还没有）
cd ~
git clone https://github.com/TNHTH/dashgo-rl-navigation.git
cd dashgo-rl-navigation

# 检查目录结构
ls -la
# 应看到: train_v2.py, dashgo_env_v2.py, README.md等文件
```

### 验证结果判断

| 测试项 | 通过 | 失败 | 解决方案 |
|--------|------|------|----------|
| Python版本 | 3.10.x | 其他版本 | 重新创建环境，指定python=3.10 |
| PyTorch导入 | 无错误 | ImportError | 检查网络连接，重新安装 |
| CUDA可用（可选） | True | False | 检查GPU驱动，重新安装CUDA |
| 项目克隆 | 有文件 | 克隆失败 | 检查网络连接或Git配置 |

**全部通过** → 可以继续阅读第二部分（环境搭建）
**有失败项** → 根据解决方案修复后再继续

---

## 1.4 项目目录导航

### 完整目录结构

```
dashgo_rl_project/                 # 项目根目录
│
├── README.md                       # 项目说明文档（644行）
├── train_v2.py                     # 训练脚本（14.8KB）
├── play.py                         # 演示脚本（7.1KB）
├── dashgo_env_v2.py               # 仿真环境定义（67.1KB）
├── dashgo_config.py               # ROS参数配置（17.3KB）
├── geo_nav_policy.py              # 轻量级网络定义（v3.1）
├── train_cfg_v2.yaml              # 训练配置（v5.0 Ultimate）
│
├── dashgo/                        # 实物ROS包（只读，严禁修改）
│   └── EAI驱动/
│       └── dashgo_bringup/config/
│           ├── my_dashgo_params.yaml  # **Sim2Real参数唯一来源**
│           └── base_local_planner_params.yaml
│
├── docs/                          # 项目文档（分类组织）
│   ├── 01-部署指南/              # Sim2Real部署相关
│   ├── 02-训练方案/              # 训练策略和超参数
│   ├── 03-问题分析/              # 问题诊断和解决方案
│   ├── 04-技术规范/              # 技术标准和规范
│   ├── 05-协议规范/              # 开发协议和流程
│   └── 06-项目历史/              # 历史记录和演变
│
├── issues/                        # 问题记录系统（70+文档）
│   ├── 2026-01-27_1730_梯度爆炸导致NaN错误.md
│   ├── 2026-01-27_1727_lidar_sensor实体不存在.md
│   └── ... (70+个问题记录)
│
├── .claude/                       # Claude AI配置
│   ├── rules/                     # 开发规则（核心）
│   │   ├── isaac-lab-development-iron-rules.md    # Isaac Lab铁律（5条）
│   │   ├── project-specific-rules.md              # 项目特定规则
│   │   └── dynamic_rules.md                        # 动态规则（23条）
│   └── skills/                    # AI技能系统
│
├── multi-agent-system/            # 智能Agent系统
│   └── agents/                    # 8个专业Agent定义
│
└── logs/                          # 训练日志（自动生成）
```

### 核心文件说明

#### 训练相关

| 文件 | 大小 | 用途 | 优先级 |
|------|------|------|--------|
| **train_v2.py** | 14.8KB | 主训练脚本 | ⭐⭐⭐ |
| **train_cfg_v2.yaml** | 2KB | 训练超参数配置 | ⭐⭐⭐ |
| **dashgo_env_v2.py** | 67.1KB | 环境定义（奖励、传感器） | ⭐⭐⭐ |
| **geo_nav_policy.py** | 8KB | 神经网络架构（v3.1） | ⭐⭐ |

#### 配置相关

| 文件 | 大小 | 用途 | 优先级 |
|------|------|------|--------|
| **dashgo_config.py** | 17.3KB | ROS参数对齐（Sim2Real） | ⭐⭐⭐ |
| **dashgo_assets.py** | 5KB | 机器人资产配置 | ⭐⭐ |
| **my_dashgo_params.yaml** | 1KB | 实物物理参数（只读） | ⭐⭐⭐ |

#### 部署相关

| 文件 | 大小 | 用途 | 优先级 |
|------|------|------|--------|
| **geo_distill_node.py** | 10KB | ROS导航节点 | ⭐⭐ |
| **safety_filter.py** | 3KB | 安全过滤器 | ⭐⭐ |

#### 文档相关

| 目录 | 用途 | 优先级 |
|------|------|--------|
| **docs/** | 项目文档（技术规范、训练方案、部署指南） | ⭐⭐ |
| **issues/** | 70+问题记录（按时间排序） | ⭐⭐⭐ |
| **.claude/rules/** | 开发规则（铁律、禁忌） | ⭐⭐⭐ |

### 关键文件快速定位

```bash
# 训练脚本
ls -lh train_v2.py  # 14.8KB

# 训练配置
ls -lh train_cfg_v2.yaml  # 2KB

# 环境定义
ls -lh dashgo_env_v2.py  # 67.1KB

# 网络架构
ls -lh geo_nav_policy.py  # 8KB

# ROS参数（Sim2Real对齐关键）
ls -lh dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml

# 问题记录（70+）
ls -lh issues/ | wc -l  # 应该显示70+个文件
```

### 目录导航技巧

```bash
# 快速跳转到核心文件
cd ~/dashgo_rl_project

# 查看训练脚本
less train_v2.py  # 按q退出

# 查看训练配置
cat train_cfg_v2.yaml

# 查看最近的问题记录
ls -lt issues/ | head -10

# 搜索特定问题
ls issues/ | grep -i "训练"
```

---

## 1.5 下一步

**恭喜！** 你已经了解了：

✅ 项目概述（技术栈、特色、目标）
✅ 系统要求（硬件、软件）
✅ 快速验证（5分钟测试）
✅ 项目目录导航

**下一部分**：环境搭建完整指南

我们将一起：
- 安装Isaac Sim 4.5
- 安装Isaac Lab
- 安装RSL-RL
- 配置所有依赖
- 验证环境完整性

**预计时间**：30-45分钟

---

**第一部分完成** | 总进度: 14% (1/7)
