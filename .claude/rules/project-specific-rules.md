# DashGo RL Navigation 项目开发规则

> **生效日期**: 2026-01-24
> **适用范围**: DashGo 机器人导航项目
> **自动加载**: 当在 `/home/gwh/dashgo_rl_project` 目录下工作时自动应用

---

## 🎯 项目概述

**项目类型**: 深度强化学习机器人导航

**🚨 开发基准（强制要求）**:
- **仿真环境**: NVIDIA Isaac Sim 4.5（严格版本锁定）
- **操作系统**: Ubuntu 20.04 LTS（严格版本锁定）
- **文档遵循**: 必须严格遵循官方相应版本的文档，严禁使用其他版本的API或示例

**核心技术栈**:
- NVIDIA Isaac Lab (基于 Isaac Sim 4.5)
- RSL-RL (强化学习库)
- Python 3.8+
- ROS Noetic (实物部署)

**项目目标**:
- 训练DashGo机器人局部导航策略
- 对齐仿真与实物参数
- 导出ONNX模型部署到实物

---

## 🔧 开发规范

### 1. 代码修改流程

```python
# 强制执行步骤
1. 修改代码
2. 运行语法检查: python -m py_compile <文件>
3. 提交到git: git add + git commit
4. 推送到GitHub: git push (如果配置了凭证)
```

**禁止行为**:
- ❌ 修改代码后不提交
- ❌ 跳过语法检查
- ❌ 覆盖 `.gitignore` 中的文件类型

### 2. 参数对齐原则

**所有参数修改必须对齐实物ROS配置**:

| 参数类别 | 配置来源 | 文件位置 |
|---------|---------|----------|
| 轮距/轮径 | `dashgo/` 文件夹 | ROS yaml文件 |
| 速度限制 | `dashgo/` 文件夹 | base_local_planner_params.yaml |
| 加速度限制 | `dashgo/` 文件夹 | base_local_planner_params.yaml |
| 机器人半径 | `dashgo/` 文件夹 | costmap_common_params.yaml |

**验证流程**:
```python
# 修改前检查
1. 读取 dashgo/ 中对应的ROS配置
2. 确认仿真参数与实物一致
3. 添加注释说明参数来源
```

### 3. Git提交规范

**Commit消息格式**:
```bash
<type>: <subject>

<body>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Type类型**:
- `feat`: 新功能
- `fix`: 修复bug
- `optimize`: 优化（参数调整、性能提升）
- `refactor`: 代码重构
- `docs`: 文档更新
- `test`: 测试相关

**示例**:
```bash
optimize: 优化Actuator配置对齐实物参数

- damping: 15.0→5.0 (降低阻尼，官方推荐5-20)
- effort_limit_sim: 10.0→20.0 Nm (提高转矩限制，留安全裕度)
- velocity_limit_sim: 8.0→5.0 rad/s ≈ 0.32 m/s (对齐ROS max_vel_x=0.3)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### 4. 文件组织规范

**项目特定规则**:
```
/home/gwh/dashgo_rl_project/
├── config/              # 配置文件（URDF、训练参数）
├── dashgo/              # 实物ROS包（只读，用于参考）
├── docs/                # 项目文档
├── logs/                # 训练日志
├── .claude/             # Claude配置（自动加载规则）
├── multi-agent-system/  # Agent定义
├── dashgo_assets.py     # Isaac Lab资产配置
├── dashgo_env_v2.py     # 训练环境
└── train_v2.py          # 训练脚本
```

**禁止操作**:
- ❌ 修改 `dashgo/` 中的文件（只读参考）
- ❌ 将 `policy.onnx`、`*.pth`、`*.ckpt` 提交到git
- ❌ 提交敏感配置（`.claude.json`、`.mcp.json`）

### 5. 官方文档优先原则（版本锁定）

**🚨 严格版本要求**:
```
- Isaac Sim 版本: 4.5（严格锁定，严禁使用其他版本）
- Ubuntu 版本: 20.04 LTS（严格锁定，严禁使用其他版本）
- 文档查询: 必须查询对应版本的官方文档
- API使用: 必须验证API在目标版本中存在
```

**核心规则**: 所有技术决策必须基于官方文档

```
决策优先级（从高到低）:
1. Isaac Lab 官方文档 (Context7查询)
2. Isaac Lab 官方示例 (GitHub: NVIDIA-Omniverse/IsaacLab)
3. RSL-RL 官方论文和文档
4. 高质量开源项目 (stars:>1000)
5. 实战经验（技术博客）
```

**执行流程**:
```python
# 修改代码前
1. 查询官方文档（Context7或官方网站）
2. 查看官方示例代码
3. 验证版本兼容性
4. 在代码中添加官方文档引用注释
```

**代码注释示例**:
```python
def compute_reward(self) -> torch.Tensor:
    """
    计算导航奖励

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    参考官方文档:
    - Isaac Sim 4.5 Reward Documentation
    - 官方示例: source/extensions/omni.isaac.lab/omni/isaac/lab/tasks/
    - 文档版本验证: 已确认API在Isaac Sim 4.5中存在

    为解决之前的原地转圈问题 (commit abc123, 2024-01-15)，
    我们移除了朝向奖励，改用势能差引导。
    """
    pass
```

---

## 🚨 禁止模式（严格禁止）

### 版本锁定相关（最高优先级）
1. **严禁使用非 Isaac Sim 4.5 版本的功能**
   - 原因：可能导致与开发环境不兼容
   - 要求：所有API必须在 Isaac Sim 4.5 文档中有记录
   - 验证：查询官方文档时明确版本号

2. **严禁使用非 Ubuntu 20.04 特有的命令或依赖**
   - 原因：部署环境不兼容
   - 要求：所有系统命令必须兼容 Ubuntu 20.04
   - 禁止：使用 Ubuntu 22.04+ 特有的包管理器或命令

3. **严禁引用其他版本的官方文档或示例**
   - 原因：API可能在不同版本间有变化
   - 要求：查询文档时必须明确指定版本
   - 示例：查询 "Isaac Sim 4.5 reward function" 而非 "Isaac Sim reward function"

### 代码相关
4. **严禁恢复朝向奖励 (Orientation Reward)**
   - 原因：会导致机器人原地转圈
   - 历史证据：commit abc123 已移除

2. **严禁大幅提高学习率**
   - 原因：训练不稳定，容易发散
   - 范围：RSL-RL 官方推荐 lr=1e-4 到 1e-3

3. **严禁使用未经验证的API**
   - 原因：可能与官方规范冲突
   - 要求：所有API必须在官方文档中有记录

### 工作流相关
1. **严禁跳过Git提交**
   - 每次修改后必须 `git add + git commit`
   - Commit消息必须清晰说明改动内容

2. **严禁覆盖 `.gitignore` 规则**
   - 训练产物（`*.pth`、`*.onnx`）不应提交
   - 敏感配置（`.claude.json`）不应提交

---

## ✅ 开发检查清单

### 代码修改前
```
□ 是否查询了官方文档？
□ 是否确认文档版本为 Isaac Sim 4.5？
□ 是否查看了官方示例？
□ 是否确认示例适用于 Isaac Sim 4.5？
□ 是否确认了参数来源（ROS配置）？
□ 是否检查了历史commit（避免重犯错误）？
□ 是否验证了功能在 Ubuntu 20.04 下可用？
```

### 代码修改时
```
□ 添加了官方文档引用注释
□ 遵循了PEP 8规范
□ 添加了类型提示
□ 处理了边界情况
□ 添加了错误处理
```

### 代码修改后
```
□ 运行了语法检查 (python -m py_compile)
□ 提交到了git (git add + git commit)
□ Commit消息清晰说明改动
□ 没有违反 `.gitignore` 规则
```

---

## 🤖 自动触发器

### 当在项目目录工作时，主AI应自动应用这些规则：

**触发条件**（满足任一即自动应用）:
- ✅ 工作目录是 `/home/gwh/dashgo_rl_project`
- ✅ 读取了项目文件（`dashgo_assets.py`、`dashgo_env_v2.py`、`train_v2.py`）
- ✅ 用户提到项目关键词（"DashGo"、"机器人导航"、"Isaac Lab"、"RSL-RL"）

**自动执行序列**:
```bash
1. Read .claude/rules/project-specific-rules.md (本文件)
2. 应用所有开发规范
3. 检查是否违反禁止模式
4. 提醒用户Git提交
```

---

## 📚 相关协议和文档

### 必读文档
- `docs/tdd-protocol.md` - 测试驱动开发协议（Backend/Frontend必须遵守）
- `docs/报告1_DashGo文件夹内容分析.md` - ROS参数分析
- `docs/报告2_项目优化方案.md` - 优化策略
- `docs/报告3_完整修改方案_代码级.md` - 具体修改步骤

### Agent文档
- `multi-agent-system/agents/robot-nav-architect.prompt.md` - 机器人导航架构师
- `multi-agent-system/shared/agent-work-principles.md` - 通用工作原则

---

## 🔧 项目特定工具配置

### Context7 查询（必须指定版本）
```bash
# Isaac Sim 4.5 官方文档（必须明确版本号）
libraryId: "/nvlabs/isaac-sim"
query: "Isaac Sim 4.5 如何定义奖励函数？官方示例是什么？"

# RSL-RL 官方文档
libraryId: "/leggedrobotics/rsl-rl"
query: "PPO算法官方实现规范 Ubuntu 20.04"
```

### GitHub 搜索
```bash
# Isaac Lab 官方示例
owner: "NVIDIA-Omniverse"
repo: "IsaacLab"
path: "source/extensions/omni.isaac.lab/omni/isaac/lab/tasks/"
```

---

**文档版本**: v1.0
**生效日期**: 2026-01-24
**维护者**: Claude Code AI System
**自动加载**: ✅ 是（在项目目录下自动应用）
