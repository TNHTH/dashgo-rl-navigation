# DashGo RL Navigation 项目完全复现指南 v5.0

> **创建时间**: 2026-01-28 22:35:00
> **第二部分**: 环境搭建完整指南
> **预计时间**: 30-45分钟
> **依赖**: 第一部分（快速开始）已完成

---

## 2.1 硬件准备

### GPU配置（必需）

**最低要求**：
- NVIDIA GTX 1660 (6GB VRAM)
- 支持：RTX系列、GTX 16系列及以上

**推荐配置**：
- NVIDIA RTX 4060 Laptop (8GB VRAM)
- 或更高性能GPU

**验证GPU**：
```bash
# 检查GPU型号
nvidia-smi --query-gpu=name --format=csv,noheader
# 预期输出（示例）: NVIDIA GeForce RTX 4060 Laptop GPU

# 检查显存大小
nvidia-smi --query-gpu=memory.total --format=csv,noheader
# 预期输出（示例）: 8192 MiB

# 检查CUDA版本
nvidia-smi
# 预期输出: CUDA Version: 12.9（或兼容版本）
```

**如果GPU检测失败**：
- ❌ 检查NVIDIA驱动是否安装：`sudo ubuntu-drivers devices`
- ❌ 重新安装驱动：`sudo apt install nvidia-driver-535`
- ❌ 重启系统：`sudo reboot`

### 内存和存储

| 组件 | 最低配置 | 推荐配置 | 检查命令 |
|------|---------|---------|---------|
| **RAM** | 16GB | 32GB | `free -h` |
| **存储** | 50GB HDD | 100GB SSD | `df -h` |

**验证**：
```bash
# 检查可用内存
free -h
# Mem: 应该 ≥ 16GB

# 检查可用存储
df -h /
# Avail: 应该 ≥ 50GB
```

---

## 2.2 Ubuntu 20.04系统配置

### 检查系统版本

```bash
# 确认是Ubuntu 20.04 LTS
lsb_release -a
# 预期输出:
# Distributor ID: Ubuntu
# Description:    Ubuntu 20.04.x LTS
# Release:        20.04
```

**如果不是Ubuntu 20.04**：
- ⚠️ 本项目不支持Ubuntu 22.04或更高版本
- ⚠️ Isaac Sim 4.5仅支持Ubuntu 20.04
- 解决方案：重新安装Ubuntu 20.04 LTS

### 安装基础依赖

```bash
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux

# 安装Python依赖
sudo apt install -y \
    python3-dev \
    python3-pip

# 验证安装
git --version    # ≥ 2.x
python3 --version # ≥ 3.8（但我们会用Conda管理Python）
```

---

## 2.3 Isaac Sim 4.5安装

### 什么是Isaac Sim？

Isaac Sim是NVIDIA开发的机器人仿真器，基于Omniverse平台。
- **物理仿真**：PhysX 5（NVIDIA自研物理引擎）
- **渲染**：RTX光线追踪
- **机器人**：URDF/USD资产支持
- **版本锁定**：本项目使用Isaac Sim 4.5（严格）

### 安装步骤（详细）

#### 步骤1: 注册NVIDIA账号（5分钟）

1. 访问：https://developer.nvidia.com/isaac-sim
2. 注册NVIDIA开发者账号（免费）
3. 下载Isaac Sim 4.5（需要登录）

**下载文件**：
- 文件名：`Isaac-Sim-4.5.0.tar.gz`
- 大小：约5GB
- 下载时间：取决于网络（10-30分钟）

#### 步骤2: 安装Isaac Sim（10分钟）

```bash
# 创建安装目录
mkdir -p ~/IsaacSim
cd ~/IsaacSim

# 解压下载的文件（替换为实际下载路径）
tar -xzf ~/Downloads/Isaac-Sim-4.5.0.tar.gz

# 设置环境变量
echo 'export ISAACSIM_PATH="$HOME/IsaacSim"' >> ~/.bashrc
source ~/.bashrc

# 验证安装
ls -la $ISAACSIM_PATH
# 应看到: isaac-sim.sh, python.sh, setup等文件
```

#### 步骤3: 验证Isaac Sim（5分钟）

```bash
# 启动Isaac Sim GUI（首次启动会慢）
cd $ISAACSIM_PATH
./isaac-sim.sh

# 预期结果：
# - Isaac Sim窗口打开
# - 可以看到机器人、物体等示例场景
# - 无错误提示

# 如果无法启动GUI（远程服务器），使用headless验证
cd $ISAACSIM_PATH
./python.sh -c "import isaacsim; print('Isaac Sim导入成功')"
```

**常见问题**：
- **问题1**: "ImportError: No module named 'isaacsim'"
  - 解决：检查`ISAACSIM_PATH`环境变量是否正确设置
  - 命令：`echo $ISAACSIM_PATH`

- **问题2**: "Cannot display GUI"
  - 解决：远程服务器需要使用headless模式
  - 命令：`./python.sh` 而非 `./isaac-sim.sh`

---

## 2.4 Isaac Lab安装与验证

### 什么是Isaac Lab？

Isaac Lab是Isaac Sim的扩展框架，专门用于强化学习。
- **强化学习环境**：OpenAI Gym/Gymnasium接口
- **算法集成**：RSL-RL、stable-baselines3等
- **版本**：0.46.4（对应Isaac Sim 4.5）

### 安装步骤

#### 步骤1: 克隆Isaac Lab（2分钟）

```bash
# 克隆仓库
cd ~
git clone https://github.com/NVIDIA-Omniverse/IsaacLab.git
cd IsaacLab

# 切换到指定版本（严格锁定）
git checkout v0.4.46  # 对应Isaac Sim 4.5

# 验证版本
git log -1 --oneline
# 应显示commit hash和v0.4.46标签
```

#### 步骤2: 安装Isaac Lab（10分钟）

```bash
# 创建专用Conda环境
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab

# 安装Isaac Lab
cd ~/IsaacLab
pip install -e .

# 验证安装
python -c "import isaaclab; print('Isaac Lab版本:', isaaclab.__version__)"
# 预期输出: Isaac Lab版本: 0.4.46
```

#### 步骤3: 配置Isaac Lab环境（5分钟）

```bash
# 添加Isaac Lab路径到环境变量
echo 'export ISAACLAB_PATH="$HOME/IsaacLab"' >> ~/.bashrc
echo 'source $ISAACLAB_PATH/isaaclab.sh' >> ~/.bashrc
source ~/.bashrc

# 验证环境变量
echo $ISAACLAB_PATH
# 预期输出: /home/你的用户名/IsaacLab
```

#### 步骤4: 运行Isaac Lab示例（5分钟）

```bash
# 激活环境
conda activate env_isaaclab

# 运行一个简单示例（headless模式）
cd ~/IsaacLab
python source/extensions/omni.isaac.lab/omni/isaac/lab/scripts/interactive_scenario.py --headless

# 预期输出：
# - [INFO] 启动Isaac Sim...
# - [INFO] 创建场景...
# - 无错误提示
```

**常见问题**：
- **问题1**: "ModuleNotFoundError: No module named 'omni.isaac.core'"
  - 解决：确保`source isaaclab.sh`在导入Isaac Lab之前执行
  - 命令：`source $ISAACLAB_PATH/isaaclab.sh`

- **问题2**: "Omniverse Kit not found"
  - 解决：检查Isaac Sim路径是否正确
  - 命令：`echo $ISAACSIM_PATH`

---

## 2.5 RSL-RL库安装

### 什么是RSL-RL？

RSL-RL是ETH Zurich开发的强化学习库，专精于四足机器人。
- **算法**：PPO（Proximal Policy Optimization）
- **优化**：GPU并行、TensorDict格式
- **版本**：v3.0.1（本项目使用）

### 安装步骤（5分钟）

```bash
# 激活Isaac Lab环境
conda activate env_isaaclab

# 克隆RSL-RL
cd ~
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl

# 安装RSL-RL
pip install -e .

# 验证安装
python -c "import rsl_rl; print('RSL-RL安装成功')"
# 预期输出: RSL-RL安装成功
```

---

## 2.6 项目依赖安装

### 安装项目特定依赖

```bash
# 激活环境
conda activate env_isaaclab

# 进入项目目录
cd ~/dashgo_rl_project

# 安装项目依赖（如果有requirements.txt）
pip install -r requirements.txt

# 或者手动安装核心依赖
pip install \
    gymnasium==1.2.0 \
    tensordict==0.9.0 \
    omegaconf==2.3.0 \
    opencv-python \
    pillow
```

### 验证项目依赖

```bash
# 创建验证脚本
cat > verify_dependencies.py << 'EOF'
#!/usr/bin/env python3
"""验证项目依赖"""

import sys

def check_package(package_name, import_name=None):
    """检查包是否可导入"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name}")
        return False

def main():
    print("=== 依赖检查 ===")

    packages = {
        "PyTorch": "torch",
        "Gymnasium": "gymnasium",
        "TensorDict": "tensordict",
        "OmegaConf": "omegaconf",
        "Isaac Lab": "isaaclab",
        "RSL-RL": "rsl_rl",
        "NumPy": "numpy",
        "OpenCV": "cv2",
    }

    success = True
    for package, import_name in packages.items():
        success &= check_package(package, import_name)

    if success:
        print("\n🎉 所有依赖已安装！")
        return 0
    else:
        print("\n⚠️ 部分依赖缺失，请重新安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# 运行验证
python verify_dependencies.py
```

---

## 2.7 环境变量配置

### 完整环境变量设置

```bash
# 编辑~/.bashrc
cat >> ~/.bashrc << 'EOF'

# === Isaac Sim & Isaac Lab ===
export ISAACSIM_PATH="$HOME/IsaacSim"
export ISAACLAB_PATH="$HOME/IsaacLab"
source $ISAACLAB_PATH/isaaclab.sh

# === 项目路径 ===
export DASHGO_PROJECT="$HOME/dashgo_rl_project"
export PYTHONPATH="$DASHGO_PROJECT:$PYTHONPATH"

# === 其他 ===
export PYTHONUNBUFFERED=1  # 确保日志实时输出
EOF

# 重新加载配置
source ~/.bashrc

# 验证环境变量
echo "Isaac Sim路径: $ISAACSIM_PATH"
echo "Isaac Lab路径: $ISAACLAB_PATH"
echo "项目路径: $DASHGO_PROJECT"
```

---

## 2.8 完整性验证脚本

### 一键验证所有配置

```bash
# 创建完整验证脚本
cat > full_verification.sh << 'EOF'
#!/bin/bash

echo "=== DashGo RL Navigation 环境完整性验证 ==="
echo ""

# 1. 检查操作系统
echo "1. 操作系统"
lsb_release -a | grep "Description"
echo ""

# 2. 检查GPU
echo "2. GPU配置"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 3. 检查Conda环境
echo "3. Conda环境"
conda env list | grep env_isaaclab
echo ""

# 4. 检查Isaac Sim
echo "4. Isaac Sim"
if [ -d "$ISAACSIM_PATH" ]; then
    echo "✅ Isaac Sim路径: $ISAACSIM_PATH"
else
    echo "❌ Isaac Sim未找到"
fi
echo ""

# 5. 检查Isaac Lab
echo "5. Isaac Lab"
if [ -d "$ISAACLAB_PATH" ]; then
    echo "✅ Isaac Lab路径: $ISAACLAB_PATH"
    conda run -n env_isaaclab python -c "import isaaclab; print('版本:', isaaclab.__version__)"
else
    echo "❌ Isaac Lab未找到"
fi
echo ""

# 6. 检查RSL-RL
echo "6. RSL-RL"
conda run -n env_isaaclab python -c "import rsl_rl; print('✅ RSL-RL已安装)" 2>/dev/null || echo "❌ RSL-RL未安装"
echo ""

# 7. 检查项目
echo "7. 项目配置"
if [ -d "$HOME/dashgo_rl_project" ]; then
    echo "✅ 项目路径: $HOME/dashgo_rl_project"
    ls -lh $HOME/dashgo_rl_project/train_v2.py
else
    echo "❌ 项目未找到"
fi
echo ""

echo "=== 验证完成 ==="
EOF

# 运行验证
chmod +x full_verification.sh
./full_verification.sh
```

### 验证结果判断

**所有项目显示✅** → 环境配置成功，可以继续第三部分
**有❌项目** → 根据错误信息修复，然后重新验证

---

## 2.9 常见问题与解决方案

### 问题1: Conda环境激活失败

**错误现象**：
```bash
conda activate env_isaaclab
# CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'
```

**解决方案**：
```bash
# 初始化conda
conda init bash

# 重启shell或执行
source ~/.bashrc

# 重新激活
conda activate env_isaaclab
```

### 问题2: Isaac Sim GUI无法启动（远程服务器）

**错误现象**：
```bash
./isaac-sim.sh
# Cannot connect to display
```

**解决方案**：
```bash
# 使用headless模式
cd $ISAACSIM_PATH
./python.sh -c "import isaacsim; print('Isaac Sim导入成功')"
```

### 问题3: Isaac Lab导入错误

**错误现象**：
```python
import isaaclab
# ModuleNotFoundError: No module named 'omni.isaac.core'
```

**解决方案**：
```bash
# 确保先source isaaclab.sh
source $ISAACLAB_PATH/isaaclab.sh

# 然后再导入Python
python -c "import isaaclab; print('成功')"
```

### 问题4: GPU显存不足

**错误现象**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 降低并行环境数量
# 在train_cfg_v2.yaml中修改
num_envs: 16  # 从256降低到16

# 或使用CPU训练（慢）
CUDA_VISIBLE_DEVICES="" python train_v2.py
```

### 问题5: PyTorch CUDA版本不匹配

**错误现象**：
```
AssertionError: Torch not compiled with CUDA enabled
```

**解决方案**：
```bash
# 检查PyTorch CUDA版本
python -c "import torch; print(torch.cuda.is_available())"

# 如果显示False，重新安装PyTorch
conda activate env_isaaclab
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 2.10 下一步

**恭喜！** 你已经完成：

✅ 硬件准备（GPU、内存、存储）
✅ Ubuntu 20.04系统配置
✅ Isaac Sim 4.5安装
✅ Isaac Lab安装与验证
✅ RSL-RL库安装
✅ 项目依赖安装
✅ 环境变量配置
✅ 完整性验证

**下一部分**：项目架构深度解析

我们将一起：
- 理解目录结构
- 分析核心代码（train_v2.py, dashgo_env_v2.py, geo_nav_policy.py）
- 理解数据流和模块依赖
- 学习关键实现细节

**预计时间**：20-30分钟

---

**第二部分完成** | 总进度: 29% (2/7)
