# MCP 工具使用规范指南

**版本**: v2.0
**更新日期**: 2026-01-20
**适用于**: 所有Agent

---

## 🔥 核心原则

### 1. 工具调用规范

#### ✅ 必须并行调用
如果工具调用之间没有依赖关系，**必须并行调用**，而不是一个一个串行调用。

**好的示例**：
```javascript
并行 [
  读取文件 "src/api/users.ts"
  读取文件 "src/components/UserList.tsx"
  搜索 "getUserById"
]
```

**不好的示例**：
```javascript
读取文件 "src/api/users.ts"
// 等待结果...
读取文件 "src/components/UserList.tsx"
// 等待结果...
搜索 "getUserById"
```

#### ✅ 推测性调用
- 如果某个文件可能有用，直接读取
- 不要等待确认
- 一次调用多个相关文件

#### ✅ 批量操作
- 一次读取多个文件，而不是一个一个
- 一次搜索多个关键词
- 一次编辑多个位置（不同文件）

---

## 📊 MCP 工具详细指南

### 🔍 Firecrawl / Tavily Search

#### 何时使用
- ✅ 市场调研、竞品分析
- ✅ 用户需求研究
- ✅ 技术趋势分析
- ✅ 实时信息查询
- ✅ 网页内容提取

#### 何时不使用
- ❌ 已经在代码中定义好的规范（用Context7）
- ❌ 简单的问答（不需要搜索）
- ❌ 已知具体文档位置（直接用GitHub工具）

#### 使用策略

**策略1: 从宽泛到具体**
```
步骤1: 搜索"项目管理工具发展趋势"（宽泛）
步骤2: 根据结果，搜索"Todo应用核心功能对比"（具体）
步骤3: 抓取竞品官网分析功能列表
```

**策略2: 多角度搜索**
```
查询1: "项目管理工具用户痛点"
查询2: "Todo app common complaints"
查询3: "task management software user reviews"
```

#### 最佳实践
1. **优先使用用户原话**：用户怎么说，你就怎么搜
2. **多轮搜索**：第一轮搜索可能不够，继续深入
3. **结合其他工具**：搜索 → Context7查方法 → 综合分析

---

### 📚 Context7

#### 何时使用
- ✅ 查询产品设计最佳实践
- ✅ 查询特定框架/库的文档
- ✅ 了解工具使用方法
- ✅ 获取代码示例

#### 何时不使用
- ❌ 实时信息（用Firecrawl/Tavily）
- ❌ 竞品分析（用Firecrawl + GitHub）
- ❌ 搜索具体代码实现（用Grep）

#### 使用示例

**✅ 好的查询**：
```
"Next.js App Router 最佳实践 2024"
"TypeScript 泛型使用指南"
"PostgreSQL vs MongoDB 选择建议"
```

**❌ 不好的查询**：
```
"Next.js"  // 太宽泛
"如何用"  // 不具体
```

#### 最佳实践
1. **包含版本号**：例如 "Next.js 14.3.0" 而不是 "Next.js"
2. **具体场景**：描述你想要做什么
3. **结合实际问题**：不要只查理论，要查实际应用

---

### 🔍 GitHub

#### 何时使用
- ✅ 搜索开源代码示例
- ✅ 查找类似项目
- ✅ 学习他人实现
- ✅ 发现流行库/工具
- ✅ 管理仓库（PR、Issue、文件）

#### 何时不使用
- ❌ 查询实时信息（用Firecrawl/Tavily）
- ❌ 查询文档（用Context7）
- ❌ 分析你自己的代码库（用Grep）

#### 使用示例

**✅ 好的搜索**：
```
"todo app React TypeScript readme"
"project management tool features"
"Next.js blog template"
```

**❌ 不好的搜索**：
```
"todo"  // 太宽泛，结果太多
"help"  // 无意义
```

#### 最佳实践
1. **搜索README文件**：通常包含项目说明和特性
2. **查看stars数**：流行度可以作为参考
3. **浏览代码示例**：学习实际实现

---

### 🌐 Chrome DevTools

#### 何时使用
- ✅ 浏览器自动化测试
- ✅ 网页交互操作
- ✅ 截图和快照
- ✅ 性能分析
- ✅ 控制台调试

#### 何时不使用
- ❌ 简单的HTTP请求（用其他工具）
- ❌ 静态网页内容获取（用Firecrawl/Tavily）

#### 核心功能

**1. 页面操作**
```javascript
- navigate_page: 导航到URL
- click: 点击元素
- fill: 填写表单
- take_snapshot: 获取页面快照
- take_screenshot: 截图
```

**2. 调试功能**
```javascript
- list_console_messages: 查看控制台消息
- list_network_requests: 查看网络请求
- evaluate_script: 执行JavaScript
```

**3. 性能分析**
```javascript
- performance_start_trace: 开始性能追踪
- performance_stop_trace: 停止并获取结果
```

#### 最佳实践
1. **先用快照理解页面结构**，再执行操作
2. **检查网络请求和错误**，确保页面加载正常
3. **使用emulate功能**测试不同设备/网络条件

---

### 💻 IDE (Integrated Development Environment)

#### 何时使用
- ✅ 执行代码（Jupyter notebook）
- ✅ 获取语言诊断信息
- ✅ 交互式开发

#### 何时不使用
- ❌ 普通文件读写（用Read/Write工具）
- ❌ 非代码环境

#### 核心功能

**1. 代码执行**
```javascript
executeCode(code): 在Jupyter内核中执行代码
```

**2. 诊断信息**
```javascript
getDiagnostics(uri?): 获取语言诊断信息
```

#### 最佳实践
1. **使用executeCode测试代码片段**，特别是数据分析
2. **检查诊断信息**，发现潜在问题
3. **代码结果会持久化**，可用于后续计算

---

### 🧠 Memory MCP

#### 何时使用
- ✅ 记录用户偏好
- ✅ 存储项目配置
- ✅ 追踪Agent性能
- ✅ 保存优化建议
- ✅ 构建知识图谱

#### 何时不使用
- ❌ 临时数据（对话中记住即可）
- ❌ 大型文档（存在文件系统）
- ❌ 频繁变化的数据（每次重新生成）

#### Memory实体类型

**1. UserProfile（用户画像）**
```json
{
  "name": "User_16907_Profile",
  "entityType": "UserProfile",
  "observations": [
    "开发风格: 质量优先",
    "代码风格: 混用模式",
    "数据库策略: 根据项目选择"
  ]
}
```

**2. TaskHistory（任务记录）**
```json
{
  "name": "Task_20260112_103000",
  "entityType": "TaskHistory",
  "observations": [
    "任务类型: Web开发",
    "耗时: 45分钟",
    "Agents: Product, Frontend, Backend",
    "技术栈: Next.js, TypeScript"
  ]
}
```

**3. AgentPerformance（Agent性能）**
```json
{
  "name": "Agent_Product_Performance",
  "entityType": "AgentPerformance",
  "observations": [
    "使用次数: 15",
    "成功率: 100%",
    "平均满意度: 4.6/5"
  ]
}
```

**4. OptimizationRules（优化规则）**
```json
{
  "name": "OptimizationRules",
  "entityType": "OptimizationRules",
  "observations": [
    "简单项目(<5文件): Product + Frontend + CodeReviewer",
    "Web项目默认技术栈: Next.js, TypeScript"
  ]
}
```

#### 最佳实践
1. **及时更新**：任务完成后立即更新Memory
2. **定期分析**：每10个任务分析一次模式
3. **清理无用数据**：删除过时的观察结果

---

### 🌐 Web Reader

#### 何时使用
- ✅ 读取网页内容（Markdown格式）
- ✅ 提取文本信息
- ✅ 简单的网页抓取

#### 何时不使用
- ❌ 需要复杂交互（用Chrome DevTools）
- ❌ 需要结构化数据提取（用Firecrawl extract）
- ❌ 实时搜索（用Tavily）

#### 核心功能
```javascript
webReader(url, options?)
  - url: 网页URL
  - return_format: "markdown" | "text"
  - timeout: 请求超时（秒）
```

#### 最佳实践
1. **优先使用markdown格式**，便于处理
2. **设置合理的超时**，避免长时间等待
3. **结合缓存使用**，提高效率

---

## 🎯 通用工具使用流程

### 信息收集流程

```
1. 明确需求
   ↓
2. 选择合适工具
   ↓
3. 制定搜索策略
   ↓
4. 并行执行多个工具调用
   ↓
5. 分析结果
   ↓
6. 必要时深入搜索
   ↓
7. 综合信息给出答案
```

### 工具选择决策树

```
需要信息？
├─ 实时/市场信息 → Firecrawl/Tavily
├─ 技术文档/最佳实践 → Context7
├─ 代码示例 → GitHub
├─ 自己的代码库 → Grep/Read
├─ 用户偏好/历史 → Memory
└─ 网页内容 → Web Reader
```

---

## ⚠️ 常见错误

### 错误1: 过度询问用户
❌ **不要**：信息不足就问用户
✅ **应该**：主动使用工具收集信息

### 错误2: 串行调用工具
❌ **不要**：一个一个调用工具
✅ **应该**：并行调用无依赖的工具

### 错误3: 搜索词太模糊
❌ **不要**：搜索"项目管理"这种宽泛词
✅ **应该**：搜索"项目管理工具核心功能对比"

### 错误4: 不深入就停止
❌ **不要**：搜索一次就停止
✅ **应该**：多轮搜索，从不同角度

### 错误5: 不验证信息
❌ **不要**：搜到什么就用什么
✅ **应该**：交叉验证，确认准确性

---

## 📋 检查清单

### 工具调用前确认
- [ ] 我是否选择了正确的工具？
- [ ] 我是否制定了好的搜索策略？
- [ ] 我是否可以并行调用多个工具？

### 工具调用后确认
- [ ] 我是否得到了有用的信息？
- [ ] 信息是否足够完整？
- [ ] 是否需要进一步搜索？

### 综合分析时确认
- [ ] 我是否从多个来源验证了信息？
- [ ] 我是否提供了完整的解决方案？
- [ ] 我是否考虑了边界情况？

---

## 🚀 优化建议

### 提升效率
1. **批量操作**：一次读取10个文件，而不是10次读取1个文件
2. **并行调用**：同时执行搜索、读取、查询
3. **缓存信息**：记住已读取的文件内容

### 提升质量
1. **多角度验证**：从不同来源确认信息
2. **深入探索**：不要停留在表面结果
3. **综合分析**：结合多个工具的结果

---

## 🔄 工具切换规则

**哪个好用用哪个**

**失败自动切换协议**：
```
工具 A 失败
  ↓
自动切换到工具 B
  ↓
完成（不问用户）
```

**示例（Git 操作）**：
```
场景1：git push 常用
1. 尝试：git push (Bash)
   ↓ 失败
2. 自动切换：mcp__github__push_files
   ↓ 完成

场景2：复杂 API 操作
1. 直接用：mcp__github__create_pull_request
   ↓ 更可靠
```

**执行方式**：
- 根据任务选择最合适的工具
- 失败后立即自动切换
- 不问用户，直接执行

---

**记住**：工具是手段，不是目的。目标是快速、准确地完成任务。

**核心原则**：
> 主动收集信息，彻底理解问题，提供完整解决方案
