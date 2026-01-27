# Evolution Mechanism - 自动规则更新机制详解

## 核心原理

当优化建议满足特定条件时，自动更新到对应的规则文件中，实现从"评估"到"应用"的闭环。

**执行方式**：AI在对话结束时模拟执行逻辑，使用Read/Write/Edit工具更新文件

---

## 三层过滤系统

### Level 1: 用户明确要求（最高优先级）

**触发条件**：
- 用户说"以后都要..."
- 用户说"记住这个..."
- 用户明确强调某个问题

**示例**：
```markdown
用户: "这次报告太长了，以后都用要点列表，不要表格"

判断结果：
- is_explicit_request = True
- action_path = "MAJOR_UPDATE"
- 立即写入规则文件
```

---

### Level 2: 频率触发（反复出现）

**触发条件**：
- 同类问题出现≥3次
- 时间跨度≤7天
- 问题模式匹配度≥80%

**示例**：
```markdown
问题追踪：
- 2026-01-14: 网页搜索失败，未使用fallback
- 2026-01-15: 网页搜索失败，未使用fallback
- 2026-01-16: 网页搜索失败，未使用fallback

判断结果：
- occurrence_count = 3
- action_path = "MAJOR_UPDATE"
- 升级为规则
```

---

### Level 3: 通用性过滤（跨对话适用）

**触发条件**：
- 建议具有普遍适用性
- 不依赖特定上下文
- 可以独立存在

**通用性判断**：
| 建议 | 是否通用 | 理由 |
|------|---------|------|
| "使用要点列表" | ✅ 是 | 适用于多选项场景 |
| "优化Python脚本" | ❌ 否 | 特定于某次对话 |
| "网页搜索用fallback" | ✅ 是 | 适用于所有搜索 |
| "减少客套话" | ✅ 是 | 适用于所有回复 |

---

## Token优化：Pre-Check协议

### Phase 1: 轻量级预检（0 Token）

在对话结束时，AI模拟执行以下逻辑：

```
问题识别 → 建议分类 → 判断是否值得写入
    ↓
排除标准（任何一条满足则只记录日志）：
├─ impact_score < 20%
├─ 任务特定问题
└─ 首次出现且用户未要求
    ↓
通过标准（任何一条满足则升级为规则）：
├─ 用户明确要求
├─ 出现次数 ≥ 3
└─ impact_score > 30% 且通用
```

---

### Phase 2: 执行（按需读取）

```
action_path == "MAJOR_UPDATE"
    ↓
1. Router: 选择目标文件
   ├─ Communication → rules/dynamic_rules.md
   ├─ Workflow → instructions.md
   └─ File Organization → rules/file-organization.md
    ↓
2. Fetch: 读取目标文件（只读1个文件）
    ↓
3. Check: 冲突检测
    ↓
4. Update: 更新文件
    ↓
5. Notify: 通知用户
```

---

## 自动分类路由

```
优化建议分析
    ↓
┌────────────────────────────────┐
│  是操作流程？                   │
│  ├─ 定义"如何执行任务"         │
│  └─ YES → instructions.md     │
└────────────────────────────────┘
    ↓ NO
┌────────────────────────────────┐
│  是行为规范？                   │
│  ├─ 定义"如何表达/沟通"        │
│  └─ YES → rules/dynamic_rules.md │
└────────────────────────────────┘
    ↓ NO
┌────────────────────────────────┐
│  是领域规则？                   │
│  ├─ 文件组织                   │
│  └─ YES → rules/file-organization.md │
└────────────────────────────────┘
    ↓ NO
记录到待审核日志
```

---

## 实施检查清单

**对话结束时**：
```
□ 统计token使用
□ 计算效率指标
□ 识别优化点
□ 执行Pre-Check（轻量级预检）
□ 判断action_path
  ├─ MINOR_LOGGING → 只记录或忽略
  └─ MAJOR_UPDATE → 执行Phase 2
    ├─ 确定目标文件
    ├─ Read工具读取文件
    ├─ 冲突检测
    ├─ Write/Edit工具更新
    └─ 通知用户
□ 清理临时文件
□ 检查规则文档存在性
```

---

## 规则更新模板

### dynamic_rules.md 规则格式（YAML）

```yaml
- id: DR-XXX
  created: 2026-01-17
  frequency: 5
  category: efficiency|security|pattern
  title: "Brief title"
  content: "Specific rule content"
  rationale: "Why this rule exists"
  impact:
    token_saving: "15%"
    error_prevention: true
  status: active|deprecated|archived
  examples:
    good: "Example of correct usage"
    bad: "Example of incorrect usage"
```

---

## 规则生命周期管理

### Health Monitoring

检查规则库健康状况（当规则数≥15时）：

```markdown
Metrics:
- Total rules: [count]
- Active rules: [count]
- Deprecated rules: [count]
- Avg. token saving: [average%]

Decision Tree:
If total_rules ≥ 20:
  → Execute merge protocol

If deprecated_rules ≥ 5:
  → Execute archive protocol
```

---

### Merge Protocol（合并相似规则）

当规则过多时，合并相似规则：

```yaml
Before:
- id: DR-001
  content: "Don't say 'okay'"
- id: DR-002
  content: "Don't say 'I'll help'"

After:
- id: DR-001
  content: "Don't use zero-information phrases (okay, I'll help, etc.)"
  merged_ids: [DR-001, DR-002]
```

**步骤**：
1. 识别相似规则
2. 合并为一个规则
3. 设置旧规则为 `status: deprecated`
4. 添加 `merged_to` 字段
5. 向用户确认

---

### Archive Protocol（归档废弃规则）

当有5个以上deprecated规则时：

**步骤**：
1. 将deprecated规则移动到 `archived_rules.md`
2. 添加归档日期和原因
3. 从 `dynamic_rules.md` 中删除

```yaml
archived_rules.md format:
- id: DR-001
  original_content: "..."
  archive_date: 2026-01-17
  archive_reason: "Merged into DR-010"
```

---

## 自动应用机制

### 技能集成

这个skill应该：
1. 自动加载到system prompt
2. 每次对话结束时自动触发
3. 生成评估报告
4. 应用到下次对话

### 提醒方式

```markdown
🤖 自动评估已运行：
- 本次对话效率：{score}/100
- 发现{count}个优化点
- 下次对话将自动应用{number}项改进

详细报告：{report_link}
```

---

## 监控指标

持续跟踪的指标：

```
平均效率得分：
- 第1周：60/100
- 第2周：70/100
- 第3周：78/100
- 第4周：85/100
- 目标：> 90/100

Token节省率：
- 第1周：15%
- 第2周：25%
- 第4周：35%
- 目标：> 40%

用户满意度：
- 首次准确率提升
- 追问次数减少
- 完成时间缩短
```

---

## 自动规则更新示例

### 示例1：用户明确要求

```markdown
对话场景：
用户：以后都用表格，不要列表

AI执行：
1. 识别：is_explicit_request = True
2. 规则起草：
   ```yaml
   - id: DR-005
     title: "表格优先"
     content: "3个以上选项使用表格展示"
     rationale: "用户明确要求"
   ```
3. 用户确认：✅
4. 更新dynamic_rules.md
```

---

### 示例2：频率触发

```markdown
问题追踪：
- 2026-01-14: 未并行调用Read工具
- 2026-01-15: 未并行调用Read工具
- 2026-01-16: 未并行调用Read工具

AI执行：
1. 识别：occurrence_count = 3
2. 规则起草：
   ```yaml
   - id: DR-006
     title: "并行Read调用"
     content: "读取多个独立文件时必须并行调用Read"
     frequency: 3
   ```
3. 用户确认：✅
4. 更新dynamic_rules.md
```

---

## 总结

**核心理念**：
> 每次对话都是学习和优化的机会

**自动执行**：
> 对话结束时自动评估，无需用户提醒

**持续改进**：
> 基于数据驱动，持续优化对话策略

**最终目标**：
> 用最少的token，提供最大的价值

---

**文档版本**: v1.0
**创建日期**: 2026-01-17
**来源**: 提取自 dialogue-optimizer v3.2 Evolution Mechanism
