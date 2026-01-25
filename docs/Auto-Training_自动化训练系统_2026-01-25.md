# DashGo Auto-Training 自动化训练系统

> **创建时间**: 2026-01-25 19:45:00
> **版本**: v1.0
> **状态**: 可用

---

## 🎯 系统概述

这是一个**AutoML式循环训练系统**，可以：
1. ✅ 自动启动训练
2. ✅ 实时监控训练进度
3. ✅ 训练结束后自动分析
4. ✅ 生成优化建议
5. ⏳ 自动修改策略并重新训练（待实现）

---

## 📁 文件结构

```
/home/gwh/dashgo_rl_project/
├── auto_train_launcher.sh      # 主启动脚本（一键启动）
├── auto_analyze.py            # 分析脚本（生成报告和建议）
├── auto_optimizer.py          # 优化脚本（待创建，自动修改参数）
└── issues/
    ├── monitoring_*.log        # 监控日志
    └── training_report_*.md    # 训练报告
```

---

## 🚀 使用方法

### 方案A：一键启动（推荐）

```bash
cd /home/gwh/dashgo_rl_project
./auto_train_launcher.sh
```

**执行流程**：
1. 清理旧日志（可选）
2. 启动训练（后台运行，PID记录）
3. 实时监控（每60秒更新一次）
4. 等待训练结束
5. 自动分析并生成报告

**输出**：
- 训练进程：`training_output.log`
- 监控日志：`issues/monitoring_YYYYMMDD_HHMMSS.log`
- 训练报告：`issues/training_report_YYYYMMDD_HHMMSS.md`

### 方案B：分步执行

#### 1. 启动训练

```bash
cd /home/gwh/dashgo_rl_project

# 启动训练（后台）
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 4096 > training_output.log 2>&1 &

# 记录PID
echo $! > train.pid
```

#### 2. 实时监控（可选）

```bash
# 实时查看训练输出
tail -f training_output.log

# 或使用监控脚本
watch -n 60 "echo 'Training Status:' && ps aux | grep train_v2.py"
```

#### 3. 训练结束后分析

```bash
python3 auto_analyze.py interactive
```

---

## 📊 监控指标

系统会自动监控以下指标：

| 指标 | 说明 | 理想值 | 警告值 |
|------|------|--------|--------|
| **Iteration** | 训练迭代次数 | 5000 | - |
| **reach_goal** | 任务完成率 | >60% | <20% |
| **Policy Noise** | 策略噪声 | <1.0 | >5.0 |
| **Mean Reward** | 平均奖励 | 正值上升 | 持续下降 |

**监控输出示例**：
```
[Iteration: 1500] Iteration: 1500 | reach_goal: 35.2% | Noise: 0.8
[Iteration: 1560] Iteration: 1560 | reach_goal: 38.1% | Noise: 0.7
```

---

## 💡 自动优化建议

系统会根据训练结果自动生成建议：

### 场景1：reach_goal率过低（<20%）

**建议**：
- 🔧 增加reach_goal奖励权重（2000.0 → 3000.0）
- 🔧 降低初始目标范围（3m → 2m）
- 🔧 增加Dense奖励权重

### 场景2：Policy Noise过高（>5.0）

**建议**：
- 🔧 增强平滑约束（action_smoothness: -0.01 → -0.02）
- 🔧 降低学习率（1.5e-4 → 1e-4）
- 🔧 检查奖励函数是否有冲突

### 场景3：reach_goal率高且Noise低（>60%, <1.0）

**建议**：
- ✅ 策略已收敛
- ✅ 可以导出ONNX模型
- ✅ 进行实物测试

---

## ⏳ 待实现功能（v2.0）

### 1. 自动修改参数

**目标**：根据分析报告自动修改配置文件

**需要实现**：
- 修改`train_cfg_v2.yaml`（超参数）
- 修改`dashgo_env_v2.py`（奖励权重）
- Git提交修改

**脚本**：`auto_optimizer.py`

```python
# 伪代码示例
def auto_optimize(suggestions):
    for suggestion in suggestions:
        if suggestion['type'] == 'reward':
            # 修改dashgo_env_v2.py中的奖励权重
            update_reward_weight('reach_goal', 3000.0)
        elif suggestion['type'] == 'learning_rate':
            # 修改train_cfg_v2.yaml中的学习率
            update_learning_rate(1e-4)

    # 提交到git
    git_commit("auto: 根据分析报告自动优化参数")
```

### 2. 循环训练

**目标**：训练→分析→优化→再训练（自动循环）

**实现方式**：
```bash
while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    # 启动训练
    ./auto_train_launcher.sh

    # 分析结果
    python3 auto_analyze.py auto

    # 检查是否需要继续优化
    if should_continue_optimization; then
        # 自动优化
        python3 auto_optimizer.py

        # 增加迭代计数
        ITERATION=$((ITERATION + 1))
    else
        # 收敛，退出
        break
    fi
done
```

### 3. TensorBoard集成

**目标**：实时可视化训练曲线

**实现方式**：
```bash
# 启动TensorBoard
tensorboard --logdir logs/ --port 6006 &

# 在浏览器中查看
# http://localhost:6006
```

---

## 🔧 高级功能

### 1. 远程监控（可选）

**方案**：使用Telegram Bot推送训练状态

**实现**：
```python
import requests

def send_telegram_notification(message):
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}

    requests.post(url, data=data)
```

### 2. 邮件报告（可选）

**方案**：训练结束后发送邮件报告

### 3. 多GPU并行训练

**方案**：同时训练多个不同参数的模型，对比效果

---

## 📋 使用检查清单

### 训练前

- [ ] 确认旧日志已清理（避免混淆）
- [ ] 确认GPU显存足够（nvidia-smi）
- [ ] 确认配置参数正确
- [ ] 设置训练轮数（train_cfg_v2.yaml）

### 训练中

- [ ] 监控GPU温度（避免过热）
- [ ] 监控显存占用
- [ ] 观察reach_goal率趋势
- [ ] 观察Policy Noise稳定性

### 训练后

- [ ] 阅读分析报告
- [ ] 查看优化建议
- [ ] 决定是否需要继续训练
- [ ] 提交到git

---

## 🚨 故障排查

### 问题1：训练启动失败

**症状**：进程立即退出

**排查**：
```bash
# 检查错误日志
tail -100 training_output.log

# 检查Python环境
python3 -c "from isaaclab.envs import mdp; print('OK')"

# 检查GPU
nvidia-smi
```

### 问题2：reach_goal率始终为0%

**可能原因**：
- 奖励函数配置错误
- threshold过严
- 观测空间有问题

**解决**：检查`dashgo_env_v2.py`中的`check_reach_goal`函数

### 问题3：Policy Noise爆炸

**症状**：Noise > 10.0

**解决**：
1. 降低学习率
2. 增加action_smoothness权重
3. 检查奖励函数是否有冲突

---

## 📚 相关文档

- Isaac Lab官方文档: https://isaac-sim.github.io/IsaacLab/
- v5.0实施方案: `docs/训练方案v5.0_最终综合版_2026-01-25.md`
- API修复记录: `issues/2026-01-25_1930_API版本不匹配_position_command_error_tanh不存在.md`

---

**创建时间**: 2026-01-25 19:45:00
**维护者**: Claude Code AI System
**版本**: v1.0
**状态**: ✅ 可用（核心功能已实现）
**下一步**: 实现自动优化功能（v2.0）
