# 基于 Cursor Agent Prompt 2.0 的改进建议

**分析日期**: 2026-01-12
**参考来源**: https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools

---

## 📊 对比总结

### Cursor Agent Prompt 2.0 的优势

| 特性 | 描述 | 价值 |
|------|------|------|
| **极其详细的工具指南** | 每个工具都有"何时使用"、"何时不使用"、大量示例 | ⭐⭐⭐⭐⭐ |
| **强调主动性** | "Autonomously resolve the query" | ⭐⭐⭐⭐⭐ |
| **彻底的信息收集** | "Be THOROUGH", "TRACE every symbol" | ⭐⭐⭐⭐⭐ |
| **严格的任务管理** | todo_write有详细使用规范 | ⭐⭐⭐⭐ |
| **代码引用格式** | 严格的CODE REFERENCES规范 | ⭐⭐⭐⭐ |
| **并行工具调用** | 明确要求并行执行 | ⭐⭐⭐⭐ |

### 你的系统的优势

| 特性 | 描述 | 价值 |
|------|------|------|
| **多Agent架构** | 8个专业Agent，分工明确 | ⭐⭐⭐⭐⭐ |
| **清晰的通信协议** | shared/contracts文件共享 | ⭐⭐⭐⭐⭐ |
| **依赖关系管理** | dependencies配置 | ⭐⭐⭐⭐ |
| **项目路由器** | project-router.json自动选择 | ⭐⭐⭐⭐⭐ |
| **技能系统** | skills目录 | ⭐⭐⭐ |

---

## 🎯 核心改进建议

### 1. **为每个Agent添加详细的工具使用指南** ⭐最重要

**当前问题**：
- Product Agent只列出了MCP工具，但没有详细的使用指导
- 没有说明何时使用哪个工具
- 没有示例和最佳实践

**改进方案**（参考Cursor格式）：

```markdown
## 🔍 Firecrawl (主要工具)

### 何时使用
- ✅ 竞品分析、市场调研
- ✅ 用户需求研究
- ✅ 技术趋势分析

### 何时不使用
- ❌ 已经在代码中定义好的规范（用Context7）
- ❌ 简单的问答（不需要搜索）
- ❌ 已知具体文档位置（直接用GitHub工具）

### 使用策略

#### 策略1: 从宽泛到具体
```
步骤1: 搜索"项目管理工具发展趋势"（宽泛）
步骤2: 根据结果，搜索"Todo应用核心功能对比"（具体）
步骤3: 抓取竞品官网分析功能列表
```

#### 策略2: 多角度搜索
```
查询1: "项目管理工具用户痛点"
查询2: "Todo app common complaints"
查询3: "task management software user reviews"
```

### 使用示例

#### ✅ 好的示例
```json
{
  "query": "2024年最佳项目管理工具功能对比",
  "explanation": "了解市场主流功能，帮助我们规划MVP",
  "sources": ["web"],
  "max_results": 10
}
```

#### ❌ 坏的示例
```json
{
  "query": "项目管理"  // 太模糊
}
```

### 最佳实践
1. **优先使用用户原话**：用户怎么说，你就怎么搜
2. **多轮搜索**：第一轮搜索可能不够，继续深入
3. **结合其他工具**：搜索 → Context7查方法 → 综合分析
```

---

### 2. **添加"主动性和彻底性"指导原则** ⭐⭐⭐⭐⭐

**当前问题**：
- Agent可能过早停止信息收集
- 没有强调自主解决问题

**改进方案**（在每个Agent的prompt末尾添加）：

```markdown
## 🚀 工作原则

### 彻底性原则 (Be THOROUGH)

在给出答案之前，确保你有**完整的上下文**：

1. **TRACE每个符号到其定义**
   - 不仅找到第一个结果
   - 继续探索替代实现
   - 考虑边界情况

2. **EXPLORE直到自信**
   - 多轮搜索，不同措辞
   - 查看多个来源
   - 交叉验证信息

3. **自主信息收集**
   - 优先使用工具而不是问用户
   - 假设用户不可用
   - 主动发现缺失的信息

### 主动性原则 (Be Proactive)

1. **不要等待确认**
   - 有合理计划就执行
   - 不要说"是否继续？"
   - 直接展示结果

2. **自主决策**
   - 在可选方案中，选择最优的
   - 明确说明选择理由
   - 不要把决策负担推给用户

3. **完整解决**
   - 不要只做一半
   - 预判下一步需求
   - 主动提供完整方案

### 信息收集检查清单

在给出最终答案前，确认：
- [ ] 我是否查看了所有相关文件？
- [ ] 我是否理解了完整的上下文？
- [ ] 我是否考虑了边界情况？
- [ ] 我是否验证了信息的准确性？
- [ ] 我是否提供了完整的解决方案？

### 示例对比

#### ❌ 不够主动
```
用户：帮我设计一个Todo应用

Agent：请问你需要哪些功能？
       你想用什么技术栈？
       目标用户是谁？
```

#### ✅ 主动且彻底
```
用户：帮我设计一个Todo应用

Agent：[自主搜索市场主流Todo应用]
     [分析核心功能]
     [给出推荐方案]

     基于市场分析，我建议实现：
     1. 核心功能：增删改查、分类标签
     2. 技术栈：Next.js + TypeScript
     3. 目标用户：个人用户

     这是最常见的配置（覆盖80%用户）。
     如果你需要团队协作功能，我可以调整方案。
```
```

---

### 3. **添加任务管理规范** ⭐⭐⭐⭐

**当前问题**：
- 没有todo使用规范
- 可能过度使用或使用不当

**改进方案**（添加到所有Agent的prompt）：

```markdown
## 📋 任务管理 (todo_write)

### 何时使用

**✅ 必须使用**：
1. 复杂多步骤任务（3+步骤）
2. 用户明确要求多个任务
3. 需要向用户展示进度
4. 接收到新的重要指令

**❌ 不要使用**：
1. 单个简单任务
2. 可以直接完成的操作
3. 纯信息查询
4. 操作性任务（linting, testing, searching）

### ⚠️ 绝对不要放入todo的内容

```
❌ "Lint代码"（这是操作，不是任务）
❌ "运行测试"（这是操作，不是任务）
❌ "搜索代码库"（这是收集信息，不是任务）
❌ "检查错误"（这是验证，不是任务）
```

**应该放入todo的内容**：
```
✅ "实现用户认证功能"（这是真正的任务）
✅ "设计数据库Schema"（这是真正的任务）
✅ "创建API端点"（这是真正的任务）
```

### 任务状态管理

1. **同时只有1个任务 in_progress**
2. **完成后立即标记为 completed**
3. **不要批量标记**：一个一个完成

### 示例

#### ✅ 好的todo使用
```
用户：实现用户注册、登录、密码重置功能

Agent创建todo：
[
  {id: "1", status: "in_progress", content: "实现用户注册API"},
  {id: "2", status: "pending", content: "实现用户登录API"},
  {id: "3", status: "pending", content: "实现密码重置API"}
]

Agent：[开始实现注册API]

Agent完成注册后：
[
  {id: "1", status: "completed", content: "实现用户注册API"},
  {id: "2", status: "in_progress", content: "实现用户登录API"},
  ...
]
```

#### ❌ 不好的todo使用
```
用户：运行测试

Agent创建todo：
[
  {id: "1", status: "in_progress", content: "运行测试"}
]

❌ 这是单个简单操作，不需要todo！
```
```

---

### 4. **添加工具调用规范** ⭐⭐⭐⭐

**当前问题**：
- 没有明确的工具调用策略
- 没有强调并行调用

**改进方案**：

```markdown
## 🔧 工具调用规范

### 并行调用优先级

**原则**：如果工具调用之间没有依赖关系，**必须并行调用**

#### ✅ 好的示例（并行）
```javascript
// 一次消息中并行调用3个工具
并行 [
  读取文件 "src/api/users.ts"
  读取文件 "src/components/UserList.tsx"
  搜索 "getUserById"
]
```

#### ❌ 坏的示例（串行）
```javascript
// 分3次调用，慢得多
读取文件 "src/api/users.ts"
// 等待结果...

读取文件 "src/components/UserList.tsx"
// 等待结果...

搜索 "getUserById"
```

### 工具调用最佳实践

1. **推测性调用**
   - 如果某个文件可能有用，直接读取
   - 不要等待确认
   - 一次调用多个相关文件

2. **批量操作**
   - 一次读取多个文件，而不是一个一个
   - 一次搜索多个关键词
   - 一次编辑多个位置

3. **避免重复调用**
   - 记住已经读取的文件内容
   - 不要重新读取相同的内容
   - 使用缓存的信息

### 工具失败处理

1. **读取文件失败**
   - 检查路径是否正确
   - 尝试使用相对路径
   - 搜索类似文件名

2. **编辑文件失败**
   - 重新读取文件
   - 确认当前内容
   - 再次尝试编辑

3. **搜索无结果**
   - 更换搜索关键词
   - 尝试更宽泛的查询
   - 使用不同的搜索策略

### 工具调用示例

#### 示例1: 批量读取相关文件
```javascript
// 需要了解用户相关的所有代码
并行 [
  读取文件 "src/api/users.ts"
  读取文件 "src/components/UserForm.tsx"
  读取文件 "src/types/user.ts"
  搜索 "interface User"
]
```

#### 示例2: 多角度搜索
```javascript
// 全面了解认证实现
并行 [
  搜索 "authentication flow"
  搜索 "login API"
  搜索 "user session"
  读取文件 "src/auth/index.ts"
]
```
```

---

### 5. **添加代码引用和编辑规范** ⭐⭐⭐

**改进方案**：

```markdown
## 📝 代码引用规范

### CODE REFERENCES（引用现有代码）

**格式**：
```
```startLine:endLine:filepath
// 代码内容
```
```

**示例**：
```12:14:app/components/Todo.tsx
export const Todo = () => {
  return <div>Todo</div>;
};
```

**规则**：
1. 必须包含：startLine, endLine, filepath
2. 不要添加语言标签
3. 至少包含1行实际代码
4. 可以用 `// ... more code ...` 截断长代码

### MARKDOWN CODE BLOCKS（新代码或提议）

**格式**：
```python
for i in range(10):
    print(i)
```

**规则**：
1. 使用标准markdown代码块
2. 添加语言标签
3. 不添加行号
4. 不使用CODE REFERENCES格式

### 代码编辑规范

1. **编辑现有文件**：使用 search_replace
   - 必须提供唯一的old_string
   - 包含3-5行上下文
   - 一次编辑一处，多处则多次调用

2. **创建新文件**：使用 write_file
   - 包含完整的文件内容
   - 添加必要的import
   - 包含文件级注释

3. **批量编辑**：
   - 不同文件：并行调用
   - 同一文件：一次调用包含所有编辑
```

---

### 6. **改进Agent协作机制** ⭐⭐⭐⭐

**当前问题**：
- Orchestrator很简陋
- 缺少上下文传递规范

**改进方案**：

```markdown
## Agent 协作规范

### 上下文传递

#### 输出格式（给下游Agent）
```yaml
# 在文件中添加明确的锚点
[[PRODUCT_OUTPUT]]
核心功能:
  - 用户认证
  - 任务管理

数据模型:
  - User: {id, email, password}
  - Task: {id, title, status, userId}

API需求:
  - POST /api/auth/register
  - POST /api/auth/login
  - GET /api/tasks
[[PRODUCT_OUTPUT_END]]
```

#### 输入读取（从上游Agent）
```markdown
## 读取上游Agent输出

首先扫描以下锚点：
- [[PRODUCT_OUTPUT]] - 产品需求
- [[ARCHITECT_OUTPUT]] - 架构设计
- [[BACKEND_OUTPUT]] - API定义

如果找不到必需的锚点，停止并请求：
"❌ 缺少上游Agent输出，请先运行[Agent名称]"
```

### 协作检查清单

每个Agent在开始工作前确认：
- [ ] 我是否读取了上游Agent的输出？
- [ ] 我是否理解了依赖关系？
- [ ] 我是否提供了清晰的输出格式？
- [ ] 我的输出是否便于下游Agent使用？

### 协作示例

```
Product Agent → 输出 [[PRODUCT_OUTPUT]]
                ↓
Architect Agent → 读取 [[PRODUCT_OUTPUT]]
               → 输出 [[ARCHITECT_OUTPUT]]
                ↓
Backend Agent → 读取 [[PRODUCT_OUTPUT]]
             → 读取 [[ARCHITECT_OUTPUT]]
             → 输出 [[BACKEND_OUTPUT]]
```
```

---

### 7. **添加错误处理和重试机制** ⭐⭐⭐

**改进方案**：

```markdown
## 🔄 错误处理和重试

### 错误处理策略

#### 1. 文件操作错误
```python
if 读取文件失败:
    尝试1: 检查路径，使用相对路径
    尝试2: 搜索相似文件名
    尝试3: 列出目录，找到正确文件
    if 全部失败:
        明确告诉用户问题，请求帮助
```

#### 2. 编辑冲突
```python
if 编辑失败（old_string不匹配）:
    重新读取文件
    查找正确的内容
    再次尝试编辑
    if 仍然失败:
        提供diff让用户选择
```

#### 3. 搜索无结果
```python
if 搜索无结果:
    尝试1: 更换搜索词
    尝试2: 使用更宽泛的查询
    尝试3: 切换搜索策略（语义→精确）
    if 全部失败:
        告诉用户搜索范围，请求指引
```

### 最大重试次数

- 读取文件：3次
- 编辑文件：3次
- 搜索操作：5次（不同关键词）

### 超时处理

- 工具调用超时：60秒
- 总任务超时：10分钟
- 超时后询问用户是否继续

### 错误消息格式

```
❌ [错误类型]: [简要描述]

**尝试的操作**: [你想要做什么]
**遇到的问题**: [具体错误信息]
**已尝试的解决方案**: [1, 2, 3]

建议:
- [方案1]
- [方案2]

是否继续？还是换个方向？
```
```

---

## 📋 实施优先级

### 🔥 高优先级（立即实施）

1. **添加工具使用详细指南** → 每个Agent的prompt
2. **添加主动性和彻底性原则** → 所有Agent
3. **添加任务管理规范** → 所有Agent

### ⚡ 中优先级（本周完成）

4. **添加工具调用规范** → 所有Agent
5. **改进Agent协作机制** → communication-protocol.md
6. **添加错误处理机制** → orchestrator.mjs

### 💡 低优先级（持续改进）

7. **代码引用规范** → 文档和训练
8. **性能优化** → orchestrator.mjs
9. **监控和日志** → 新增功能

---

## 📝 具体文件修改清单

### 需要修改的文件

```
multi-agent-system/
├── agents/
│   ├── product.prompt.md       [修改]
│   ├── architect.prompt.md     [修改]
│   ├── frontend.prompt.md      [修改]
│   ├── backend.prompt.md       [修改]
│   ├── code-reviewer.prompt.md [修改]
│   ├── devops.prompt.md        [修改]
│   └── docs.prompt.md          [修改]
│
├── shared/
│   ├── communication-protocol.md   [修改]
│   ├── tool-guidelines.md          [新增]
│   └── task-management.md          [新增]
│
├── orchestrator.mjs            [修改]
└── README.md                    [更新]
```

---

## 🎯 总结

**你的系统已经很好**，主要改进方向是：

1. **更详细的工具使用指南** → 让Agent更智能地使用MCP工具
2. **更强的主动性** → 减少用户决策负担
3. **更彻底的信息收集** → 提供更完整的解决方案
4. **更规范的协作** → Agent之间配合更默契

**这些都是从Cursor的实践中总结的最佳实践**，可以显著提升你的Agent系统质量。

---

**参考来源**：
- Cursor Agent Prompt 2.0 (38KB)
- Cursor Agent Tools v1.0 (23KB)
- 你的multi-agent-system
