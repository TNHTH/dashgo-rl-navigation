# 任务评估模板

用于 UserTracker Agent 在任务完成后进行系统化评估。

---

## 快速评估（自动填充）

### 基础信息
```markdown
## 📋 任务基础信息

- **任务ID**: {{TASK_ID}}
- **任务类型**: [开发/分析/重构/部署/学习/其他]
- **任务描述**: {{DESCRIPTION}}
- **复杂度**: [简单/中等/复杂]

### 时间记录
- **开始时间**: {{START_TIME}}
- **结束时间**: {{END_TIME}}
- **实际耗时**: {{ACTUAL_DURATION}} 分钟
- **预期耗时**: {{ESTIMATED_DURATION}} 分钟
- **时间偏差**: {{TIME_DEVIATION}}% (正数=超时，负数=提前)

### Agent 使用
- **启动的 Agents**: {{AGENTS_USED}}
- **Agent 执行顺序**: {{EXECUTION_ORDER}}
- **并行/串行**: {{EXECUTION_MODE}}
- **Agent 切换次数**: {{SWITCH_COUNT}}
```

---

## 详细评估

### 1. 效率评估 (A/B/C/D)

#### 时间效率
**评分标准**：
- **A (优秀)**: 实际耗时 ≤ 预期耗时，无明显返工
- **B (良好)**: 实际耗时 ≤ 预期耗时 × 1.2
- **C (一般)**: 实际耗时 ≤ 预期耗时 × 1.5
- **D (需改进)**: 实际耗时 > 预期耗时 × 1.5

**本次评分**: {{TIME_EFFICIENCY_SCORE}}

**分析**:
```
✅ 做得好的地方:
   - [哪些环节特别高效？]
   - [哪些 Agent 协作顺畅？]

⚠️  瓶颈识别:
   - [哪些环节耗时过长？]
   - [哪些 Agent 响应慢？]
   - [是否有等待时间？]
```

#### 流程效率
```
- Agent 选择合理性: {{AGENT_SELECTION_SCORE}}/5
- 信息传递效率: {{INFO_TRANSFER_SCORE}}/5
- 决策效率: {{DECISION_EFFICIENCY_SCORE}}/5
- 返工次数: {{REWORK_COUNT}}

**流程顺畅度**: {{FLOW_SMOOTHNESS}}/5
```

---

### 2. 质量评估 (1-5分)

#### 结果质量
```
**功能完整性**: {{COMPLETENESS}}/5
- 是否实现了所有需求？
- 是否有遗漏？

**代码质量** (如适用): {{CODE_QUALITY}}/5
- Code Reviewer 发现的问题数: {{ISSUES_FOUND}}
- 严重问题: {{CRITICAL_ISSUES}}
- 主要问题: {{MAJOR_ISSUES}}
- 次要问题: {{MINOR_ISSUES}}

**准确性**: {{ACCURACY}}/5
- 是否有错误？
- 是否需要多次修正？
```

#### 用户满意度
```
**整体满意度**: {{SATISFACTION}}/5

**具体反馈**:
- ✅ [最满意的 1-2 个点]
- ⚠️  [最不满意的 1-2 个点]
- 💡 [改进建议]
```

---

### 3. 成本效益分析

#### Token 消耗
```
**总 Token 消耗**: {{TOTAL_TOKENS}}
**预估成本**: {{ESTIMATED_COST}} USD

各 Agent Token 占比:
{{AGENT_TOKEN_BREAKDOWN}}

**成本效益比**: {{COST_BENEFIT}}
- [高/中/低]
- 人工做同样任务需要: {{MANUAL_TIME}} 分钟
- 节省时间: {{TIME_SAVED}} 分钟
- 时间价值: {{TIME_VALUE}}
```

---

### 4. Agent 性能评估

#### 各 Agent 表现
```
Agent: Product Manager
- 启动时间: {{PRODUCT_START}}
- 完成质量: {{PRODUCT_QUALITY}}/5
- Token 消耗: {{PRODUCT_TOKENS}}
- 评价: {{PRODUCT_COMMENT}}

Agent: Architect
- 启动时间: {{ARCHITECT_START}}
- 完成质量: {{ARCHITECT_QUALITY}}/5
- Token 消耗: {{ARCHITECT_TOKENS}}
- 评价: {{ARCHITECT_COMMENT}}

[... 其他 Agents]
```

#### 协作效果
```
**Agent 间协作**: {{COLLABORATION}}/5
- 信息传递准确性: {{INFO_ACCURACY}}
- 依赖关系处理: {{DEPENDENCY_HANDLING}}
- 并行执行效率: {{PARALLEL_EFFICIENCY}}
```

---

### 5. 用户习惯观察

#### 决策模式
```
本次任务观察到的用户行为:
- [ ] 用户主动询问细节
- [ ] 用户让 AI 自主决策
- [ ] 用户中途改变需求
- [ ] 用户提供明确的约束条件
- [ ] 用户要求多次修改

**决策风格倾向**: [快速决策/充分思考/混合]
```

#### 沟通模式
```
- 沟通频率: [高频/中频/低频]
- 反馈风格: [详细/简洁/无反馈]
- 提问倾向: [主动提问/被动接受/混合]
```

#### 技术偏好
```
- 技术栈选择: [用户指定/AI推荐/混合]
- 代码风格要求: [严格/灵活/无要求]
- 质量标准: [高/中/低]
```

---

## 学习与优化

### 关键发现

#### ✅ 成功经验
```
1. [这次做得特别好的地方]
2. [值得重复的配置]
3. [用户特别满意的环节]
```

#### ⚠️ 改进空间
```
1. [这次遇到的问题]
2. [可以优化的地方]
3. [用户不满意的环节]
```

#### 🔍 模式识别
```
与历史任务对比:
- 这次与 [任务ID] 相似，但效率提升了 X%
- 用户仍然偏好 [某种方式]
- 检测到新的模式: [描述]
```

---

### 优化建议

#### Agent 优化
```
建议下次调整:
- [添加/移除/替换] 哪些 Agent?
- 调整 Agent 启动顺序?
- 改变并行/串行策略?
```

#### 流程优化
```
- 可以跳过的环节: [列表]
- 需要增加的环节: [列表]
- 顺序调整建议: [列表]
```

#### 预期优化
```
基于本次经验，预计下次类似任务:
- 可节省时间: {{ESTIMATED_SAVINGS}} 分钟
- 可提升质量: {{QUALITY_IMPROVEMENT}}%
- 可降低成本: {{COST_REDUCTION}}%
```

---

## 用户画像更新

### 本次任务带来的变化

```markdown
## 用户画像变更

**更新前**: [描述之前的特征]
**更新后**: [描述新的特征]

**变化原因**: [为什么会有这个变化]
**置信度**: [高/中/低]
```

### 统计数据更新

```markdown
## 统计数据

**任务统计**:
- 已完成任务总数: {{TOTAL_TASKS}}
- 本周完成任务: {{WEEKLY_TASKS}}
- 本月完成任务: {{MONTHLY_TASKS}}

**效率统计**:
- 平均耗时: {{AVG_DURATION}} 分钟
- 时间效率评分: {{AVG_EFFICIENCY}}
- 平均满意度: {{AVG_SATISFACTION}}/5

**Agent 使用统计**:
- 最常用的 Agent: {{MOST_USED_AGENT}}
- 使用次数: {{USAGE_COUNT}}
- 成功率: {{SUCCESS_RATE}}

**技术栈偏好**:
- 常用语言: {{TOP_LANGUAGES}}
- 常用框架: {{TOP_FRAMEWORKS}}
- 代码风格: {{CODE_STYLE}}
```

---

## 自动更新建议

### 需要立即更新的内容

```python
# 更新 UserProfile
await update_user_profile({
    "total_tasks": get_total_tasks() + 1,
    "last_task_efficiency": efficiency_score,
    "avg_satisfaction": calculate_avg_satisfaction()
})

# 更新 AgentPerformance
for agent in used_agents:
    await update_agent_performance(agent, {
        "usage_count": get_usage_count(agent) + 1,
        "avg_tokens": calculate_avg_tokens(agent)
    })

# 更新 OptimizationRules
if should_update_rules(task_count):
    await generate_new_rules()
```

### 需要分析的模式

```python
# 分析效率模式
efficiency_patterns = analyze({
    "什么时间效率最高？",
    "哪些 Agent 组合最有效？",
    "串行 vs 并行，哪个更快？"
})

# 分析偏好变化
preference_changes = detect({
    "技术栈偏好是否改变？",
    "沟通风格是否变化？",
    "质量 vs 速度权衡是否调整？"
})
```

---

## 报告生成

### 简洁报告（展示给用户）

```markdown
## 📊 任务完成报告

✅ **任务**: {{TASK_DESCRIPTION}}
⏱️ **耗时**: {{DURATION}} 分钟
⭐ **满意度**: {{SATISFACTION}}/5

**关键亮点**:
- {{HIGHLIGHT_1}}
- {{HIGHLIGHT_2}}

**优化空间**:
- {{IMPROVEMENT_1}}

下次建议:
- {{NEXT_TIME_TIP}}
```

### 详细报告（存储到 Memory）

```markdown
## 📋 详细评估报告 v[VERSION]

[包含上述所有内容]

**生成时间**: {{GENERATED_AT}}
**任务ID**: {{TASK_ID}}
```

---

## 使用流程

1. **任务开始时**
   - 记录开始时间
   - 识别任务类型
   - 预估复杂度

2. **任务进行中**
   - 静默记录关键决策
   - 观察 Agent 调用情况
   - 注意用户行为模式

3. **任务完成后**
   - 立即生成简洁报告
   - 后台生成详细报告
   - 更新所有统计数据

4. **每 N 个任务后**
   - 进行深度分析
   - 生成优化建议
   - 更新用户画像

---

## Memory 实体结构（4个核心）

### 1. UserProfile（用户画像）
**用途**: 存储用户的整体特征和统计数据
**更新频率**: 每5个任务

```json
{
  "name": "User_16907_Profile",
  "entityType": "UserProfile",
  "observations": [
    "开发风格: 质量优先",
    "代码风格: 混用模式",
    "数据库策略: 根据项目选择",
    "总任务数: 0",
    "平均耗时: 0分钟",
    "平均满意度: 0/5"
  ]
}
```

### 2. TaskHistory（任务记录）
**用途**: 记录每个任务的详细信息
**更新频率**: 每个任务创建新实体

```json
{
  "name": "Task_[时间戳]",
  "entityType": "TaskHistory",
  "observations": [
    "任务类型: [开发/分析/重构/部署]",
    "任务描述: [简短描述]",
    "耗时: X分钟",
    "Agents: [列表]",
    "技术栈: [列表]",
    "效率评分: [A/B/C/D]",
    "质量评分: X/5",
    "用户满意度: X/5"
  ]
}
```

### 3. AgentPerformance（Agent性能）
**用途**: 追踪每个Agent的使用情况和表现
**更新频率**: 每个任务更新

```json
{
  "name": "Agent_[Agent名称]_Performance",
  "entityType": "AgentPerformance",
  "observations": [
    "使用次数: X",
    "成功率: X%",
    "平均满意度: X/5",
    "平均Token消耗: X",
    "最佳场景: [描述]",
    "不推荐场景: [描述]"
  ]
}
```

### 4. OptimizationRules（优化规则）
**用途**: 存储基于历史数据生成的优化建议
**更新频率**: 每10个任务

```json
{
  "name": "OptimizationRules",
  "entityType": "OptimizationRules",
  "observations": [
    "简单项目(<5文件): Product + Frontend + CodeReviewer",
    "复杂项目: 全部Agents",
    "质量优先: 必须包含CodeReviewer",
    "用户常用技术栈: Next.js, TypeScript",
    "高效时段: 上午9-11点"
  ]
}
```

---

## 自动化工作流程

### 日常对话（后台静默记录）
```python
# 主Agent在对话中自动记录
记录内容:
- 任务类型和描述
- 启动的Agents
- 用户的选择和偏好
- 技术栈选择
- 开始时间
```

### 任务完成时（自动触发评估）
```python
# 检测到任务完成信号
触发条件:
- 用户说"完成""好了""可以了"
- 用户说"谢谢"或表示感谢
- 长时间无新消息
- 开始新话题

自动执行:
1. 读取本模板（task-evaluation-template.md）
2. 计算效率和质量评分
3. 创建 TaskHistory 实体
4. 更新 UserProfile 统计
5. 更新 AgentPerformance 数据
6. 如果任务数 % 10 == 0: 更新 OptimizationRules
7. 生成简短报告（可选展示）
```

### 每10个任务（自动深度分析）
```python
# 自动触发深度分析
分析内容:
1. 读取最近10个 TaskHistory
2. 识别模式（技术栈偏好、Agent使用频率）
3. 计算效率趋势
4. 生成优化建议
5. 更新 OptimizationRules
6. 展示深度分析报告
```

### 下次任务（自动应用学习）
```python
# 在开始新任务前自动读取Memory
应用内容:
1. 从 UserProfile 读取用户偏好
2. 从 OptimizationRules 读取优化建议
3. 自动推荐熟悉的工具
4. 自动调整沟通方式
5. 自动优化Agent选择
```

---

## Memory 操作示例

### 创建任务记录
```python
await mcp__memory__create_entities({
    "entities": [{
        "name": f"Task_{timestamp}",
        "entityType": "TaskHistory",
        "observations": [
            f"任务类型: {task_type}",
            f"耗时: {duration}分钟",
            f"Agents: {', '.join(agents_used)}",
            f"技术栈: {', '.join(tech_stack)}",
            f"效率评分: {efficiency_grade}",
            f"质量评分: {quality_score}/5"
        ]
    }]
})
```

### 更新用户画像
```python
await mcp__memory__add_observations({
    "observations": [{
        "entityName": "User_16907_Profile",
        "contents": [
            f"总任务数: {total_tasks + 1}",
            f"平均耗时: {avg_duration}分钟",
            f"平均满意度: {avg_satisfaction}/5"
        ]
    }]
})
```

### 更新Agent性能
```python
for agent in agents_used:
    await mcp__memory__add_observations({
        "observations": [{
            "entityName": f"Agent_{agent}_Performance",
            "contents": [
                f"使用次数: {usage_count + 1}",
                f"平均满意度: {new_avg_satisfaction}/5"
            ]
        }]
    })
```

---

**模板版本**: v2.0
**最后更新**: 2026-01-12
**维护者**: 主Agent（自动应用）
**新增**: 4实体Memory结构 + 自动化流程
