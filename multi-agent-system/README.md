# 智能多 Agent 协作系统 - 最终整合版

一个让 AI 自动识别项目类型并协调多个专家 Agent 并行协作开发项目的完整系统。整合了 v1 和 v2 的所有精华特性。

## 核心特性

### v2 特性（现代化开发体验）
- **自动项目识别** - 根据描述自动判断项目类型（Web/CLI/API/移动端等）
- **智能路由** - `/dev` skill 自动调用最合适的 Agent 组合
- **锚点机制** - 使用 `[[TAG]]` 自动传递上下文，无需手动管理文件
- **命名权威** - Architect Agent 统一管理所有命名规范
- **MCP 工具集成** - Context7、GitHub、Firecrawl、Chrome DevTools
- **详细 Prompt** - backend/frontend 有完整代码示例和模板

### v1 特性（可靠的质量保障）
- **代码审查 Agent** - 自动检查代码质量、安全性、性能
- **文档 Agent** - 独立的文档编写专家
- **共享契约** - Agent 间通过 `shared/` 目录通信
- **可选编排器** - orchestrator.mjs 作为备选编排方案

## 架构概览

```
用户输入项目想法
        │
        ▼
┌─────────────────────────────────────┐
│       /dev Skill (智能路由器)        │
│  ┌──────────────────────────────┐  │
│  │ 1. 识别项目类型               │  │
│  │ 2. 生成 project_master.md    │  │
│  │ 3. 确定需要的 Agents          │  │
│  │ 4. 构建执行计划               │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│         Agent 协作执行               │
│  ┌────────┐  ┌────────┐  ┌────────┐│
│  │Product │→│Architect│→│Backend ││ 第 1 轮
│  └────────┘  └────────┘  └────────┘│
│  ┌────────┐  ┌────────┐  ┌────────┐│
│  │Frontend│  │ DevOps │  │Code Review││ 第 2 轮
│  └────────┘  └────────┘  └────────┘│
│  ┌────────┐                     │
│  │  Docs  │                     │ 第 3 轮
│  └────────┘                     │
└─────────────────────────────────────┘
        │
        ▼
    项目完成 + 文档 + 质量报告
```

## 支持的项目类型

| 类型 | 触发关键词 | Agents | 技术栈 |
|------|-----------|--------|--------|
| **Web 应用** | 网站、web、前端、后端、管理系统、dashboard、admin、crud | product, architect, backend, frontend, devops, code-reviewer, docs | React + Node.js + PostgreSQL |
| **CLI 工具** | cli、命令行、工具、脚本、自动化、文件处理、爬虫 | product, architect, backend, devops, code-reviewer, docs | Python/Node.js + Click |
| **API 服务** | api、接口、后端、微服务、restful、graphql | product, architect, backend, devops, code-reviewer, docs | Express/Fastify + PostgreSQL |
| **桌面应用** | 桌面、exe、应用、软件、windows、mac、electron、tauri | product, architect, frontend, backend, devops, code-reviewer, docs | Electron + React |
| **移动应用** | app、手机、android、ios、移动端、react native、flutter、小程序 | product, architect, frontend, devops, code-reviewer, docs | React Native/Flutter |
| **数据分析** | 数据分析、可视化、图表、报表、pandas、jupyter | product, architect, backend, code-reviewer, docs | Python + Pandas + Plotly |
| **ML/AI** | 机器学习、ai、模型、训练、深度学习、tensorflow、pytorch | product, architect, backend, code-reviewer, docs | PyTorch/TensorFlow |

## Agent 职责

| Agent | 角色 | 主要职责 | MCP 工具 |
|-------|------|---------|---------|
| **Product** | 产品经理 | 需求分析、功能拆解、用户故事、MVP 规划 | Firecrawl, Context7, GitHub |
| **Architect** | 架构师 | 技术选型、系统设计、数据库设计、**命名权威** | Context7, GitHub, Firecrawl |
| **Backend** | 后端工程师 | API 实现、业务逻辑、数据操作、完整代码 | Context7, GitHub |
| **Frontend** | 前端工程师 | UI 实现、状态管理、API 集成、组件开发 | Context7, Chrome DevTools, GitHub |
| **DevOps** | 运维工程师 | Docker 容器化、CI/CD、部署配置、监控 | Context7, GitHub |
| **Code Reviewer** | 代码审查 | 代码质量、安全性、性能、最佳实践检查 | Context7, GitHub |
| **Docs** | 文档工程师 | README、API 文档、架构文档、使用指南 | Context7, GitHub |

## 快速开始

### 方式1: 使用 /dev Skill（推荐）

**1. 安装 Skill**

将 `.claude/skills/dev.json` 复制到你的项目：

```bash
cp -r multi-agent-system/final/.claude /your-project/.claude
```

**2. 使用方法**

在 Claude Code 中输入：

```
/dev 我想做一个任务管理应用，用户可以创建、编辑、删除任务
```

系统会自动：
1. 识别这是一个**全栈 Web 应用**
2. 生成 `project_master.md`（项目宪法）
3. 并行启动以下 Agents：
   - Product Manager（需求分析）
   - Architect（架构设计）
   - Backend（API 实现）
   - Frontend（UI 实现）
   - DevOps（部署配置）
   - Code Reviewer（代码审查）
   - Docs（文档编写）

### 方式2: 使用配置文件（传统方式）

**1. 配置项目**

复制并修改 `project.config.example.yaml`：

```yaml
project:
  name: "我的项目"
  description: "项目描述"

agents:
  product:
    name: "产品经理"
    responsibilities:
      - "需求分析"
      - "功能拆解"
    dependencies: []

  architect:
    name: "架构师"
    dependencies: [product]
    # ... 其他配置
```

**2. 执行开发**

```bash
# 使用编排器（可选）
node orchestrator.mjs project.config.yaml
```

## MCP 工具配置

每个 Agent 配置了最合适的 MCP 服务器：

### MCP 服务器能力

| MCP 服务器 | 功能 | 使用场景 |
|-----------|------|---------|
| **Context7** | 查询最新库文档 | 获取框架 API、最佳实践、代码示例 |
| **GitHub** | 代码搜索和仓库操作 | 查找实际代码示例、开源项目案例 |
| **Firecrawl** | 网页搜索和内容抓取 | 市场调研、竞品分析、用户研究 |
| **Chrome DevTools** | 浏览器自动化和调试 | 前端调试、响应式测试、性能分析 |

### Agent → MCP 映射

```
Product:  Firecrawl → Context7 → GitHub
          (市场调研) → (产品方法) → (案例参考)

Architect: Context7 → GitHub → Firecrawl
          (技术文档) → (实际案例) → (架构文章)

Backend:   Context7 → GitHub
          (框架文档) → (代码示例)

Frontend:  Context7 → Chrome DevTools → GitHub
          (框架文档) → (调试验证) → (UI 组件)

DevOps:    Context7 → GitHub
          (工具文档) → (配置示例)

Code Reviewer: Context7 → GitHub
              (最佳实践) → (安全检查)

Docs:      Context7 → GitHub
          (文档规范) → (优秀案例)
```

**详细配置请参考**: [MCP-GUIDE.md](./MCP-GUIDE.md)

## 工作流程详解

### 第一阶段：项目初始化

你输入想法后，系统会生成 `project_master.md`：

```markdown
# Project Master: Todo App

[[PROJECT_GENESIS]]
---
**Project Name**: Todo App
**Project Type**: 全栈 Web 应用
**Tech Stack Strategy**: React + Node.js + PostgreSQL + Prisma
**Global Rules**:
1. Strict TypeScript
2. camelCase for vars, PascalCase for components
3. Feature-based folder structure
4. All files named by Architect Agent
---
[[PROJECT_GENESIS_END]]

## File Structure Map
project/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   ├── models/
│   │   ├── services/
│   │   └── types/
│   └── prisma/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── api/
│   └── public/
├── shared/
│   ├── types/
│   ├── contracts/
│   └── schemas/
└── docs/

## API Contract
[[API_CONTRACT]]
{
  "endpoints": [
    {
      "method": "GET",
      "path": "/api/todos",
      "response": { "data": [...] }
    }
  ]
}
[[API_CONTRACT_END]]

## Active Features
- [x] 需求分析 (Product Agent)
- [x] 架构设计 (Architect Agent)
- [ ] API 实现 (Backend Agent)
- [ ] UI 实现 (Frontend Agent)
- [ ] 部署配置 (DevOps Agent)
- [ ] 代码审查 (Code Reviewer Agent)
- [ ] 文档编写 (Docs Agent)
```

### 第二阶段：并行开发

系统会自动启动 Agents，根据依赖关系决定并行或串行：

**Web 应用示例**：
```
第 1 轮 (并行): Product + Architect
    ↓
第 2 轮 (并行): Backend + Frontend + DevOps
    ↓
第 3 轮 (并行): Code Reviewer + Docs
```

**CLI 工具示例**：
```
第 1 轮: Product
    ↓
第 2 轮: Architect
    ↓
第 3 轮: Backend
    ↓
第 4 轮: DevOps + Code Reviewer + Docs
```

每个 Agent 会：
1. 扫描对话历史找到 `[[TAG]]` 锚点
2. 读取需要的上下文
3. 执行任务
4. 输出结果并更新 `project_master.md`

### 第三阶段：质量保障（Code Reviewer）

Code Reviewer Agent 会：

1. **自动扫描**所有生成的代码
2. **检查**：
   - 代码质量（可读性、可维护性、复杂度）
   - 安全性（SQL 注入、XSS、敏感信息泄露）
   - 性能（数据库查询、缓存策略）
   - 最佳实践（设计模式、错误处理、测试覆盖）
   - 架构一致性（模块化、依赖关系）
3. **输出报告**：
   - 严重问题（Critical）- 必须修复
   - 主要问题（Major）- 应该修复
   - 次要问题（Minor）- 建议修复
4. **提供修复建议**和示例代码

### 第四阶段：结果整合

系统会生成最终的项目报告：

```markdown
# 执行报告

## 项目概况
- 类型: 全栈 Web 应用
- 技术栈: React + Node.js + PostgreSQL
- 文件数: 25
- 代码行数: ~3500

## 已完成功能
✅ 需求分析
✅ 架构设计
✅ API 实现
✅ UI 实现
✅ 部署配置
✅ 代码审查
✅ 文档生成

## 代码审查报告
- 评分: 8.5/10
- 严重问题: 0
- 主要问题: 2
- 次要问题: 5

## 项目文件
backend/src/api/todos.ts
backend/src/models/Todo.ts
frontend/src/components/TodoList.tsx
docker-compose.yml
README.md
...

## 下一步建议
1. 运行 `npm install` 安装依赖
2. 配置 `.env` 文件
3. 运行 `docker-compose up` 启动服务
4. 访问 http://localhost:3000
5. 运行测试套件
```

## Agent 通信机制

Multi-Agent 系统支持三种 Agent 间通信方式：

### 1. 锚点机制（主要方式，v2 特性）

使用 `[[TAG]]` 格式在对话历史中标记上下文：

```markdown
[[PROJECT_GENESIS]]
项目宪法内容
[[PROJECT_GENESIS_END]]

[[ATOMIC_PRD]]
产品需求文档
[[ATOMIC_PRD_END]]

[[ARCHITECTURE_DESIGN]]
架构设计文档
[[ARCHITECTURE_DESIGN_END]]

[[API_CONTRACT]]
{
  "endpoints": [...]
}
[[API_CONTRACT_END]]
```

**优点**：
- 无需手动管理文件
- Agent 自动扫描对话历史找到上下文
- 适合 AI 对话场景

### 2. 文件共享（辅助方式，v1 特性）

Agent 通过读写共享目录中的文件进行通信：

```
shared/
├── types/           # TypeScript 类型定义
│   └── api.ts
├── schemas/         # 数据模型 Schema
│   └── user.json
├── contracts/       # API 契约
│   └── openapi.yaml
└── config/          # 共享配置
    └── env.template.json
```

**流程**：
1. Backend Agent 写入 `shared/contracts/backend-api.yaml`
2. Frontend Agent 读取并生成客户端代码
3. Docs Agent 收集所有文件生成文档

**优点**：
- 持久化存储
- 支持版本控制
- 便于人工审查

### 3. 消息传递（编排器方式）

通过 Orchestrator 中转结构化消息：

```json
{
  "from": "backend",
  "to": "frontend",
  "type": "info",
  "payload": {
    "message": "API 已完成",
    "apiSpecPath": "./shared/contracts/backend-api.yaml"
  }
}
```

**适用场景**：
- 使用 orchestrator.mjs 时
- 需要 Agent 间直接通信
- 复杂的依赖关系

## 高级用法

### 自定义技术栈

在 `project_master.md` 中修改技术栈：

```yaml
Tech Stack Strategy:
  frontend: SvelteKit
  backend: Go + Fiber
  database: MongoDB
  orm: -  # MongoDB 不需要 ORM
```

### 添加自定义 Agent

**步骤 1**: 创建 Agent Prompt

在 `agents/custom-agent.prompt.md`：

```markdown
# Custom Agent

你是...

## 你的职责
1. ...
2. ...

## 可用的 MCP 工具
...
```

**步骤 2**: 在 project-router.json 中注册

```json
{
  "id": "custom-app",
  "name": "自定义应用",
  "patterns": ["关键词1", "关键词2"],
  "skill": "/custom-app",
  "agents": ["product", "architect", "custom-agent", "docs"]
}
```

### 自定义执行策略

在 `agent-mcp-config.json` 中修改：

```json
{
  "executionStrategy": {
    "mode": "smart-parallel",
    "rules": [
      "简单项目（<5个文件）：串行执行",
      "中等项目（5-20个文件）：混合模式",
      "复杂项目（>20个文件）：并行最大化"
    ]
  }
}
```

## 代码审查详细说明

Code Reviewer Agent 提供全面的代码质量检查：

### 审查维度

**1. 代码质量**
- 可读性和可维护性
- 代码异味（Code Smells）
- 命名规范一致性
- 圈复杂度控制

**2. 安全性**
- SQL 注入防护
- XSS/CSRF 防护
- 敏感信息泄露检查
- 依赖包漏洞扫描

**3. 性能**
- 数据库查询优化（N+1 问题）
- 缓存策略检查
- 内存使用优化
- 并发处理验证

**4. 最佳实践**
- 设计模式使用
- 错误处理完善性
- 测试覆盖率
- 文档完整性

**5. 架构**
- 模块化和解耦
- 循环依赖检测
- 接口设计合理性
- 前后端契约一致性

### 审查报告格式

```markdown
# Code Review Report

## 总体评估
- 评分: 8.5/10
- 审查文件: [文件列表]
- 审查时间: [时间戳]

## 问题统计
- 严重问题: 0 个
- 主要问题: 2 个
- 次要问题: 5 个

## 详细问题列表

### 🔴 严重问题
(无)

### 🟡 主要问题
1. **缺少输入验证**
   - 文件: `backend/src/api/todos.ts:45`
   - 描述: POST /api/todos 未验证必填字段
   - 风险: 可能导致数据库异常
   - 建议:
     ```typescript
     if (!title || !description) {
       return res.status(400).json({ error: 'Missing required fields' });
     }
     ```

2. **SQL 注入风险**
   - 文件: `backend/src/services/userService.ts:23`
   - 描述: 直接拼接 SQL 查询
   - 风险: 可能被注入攻击
   - 建议: 使用 Prisma 参数化查询

### 🟢 次要问题
...

## 最佳实践建议
1. 添加 API 速率限制
2. 实现请求日志记录
3. 使用环境变量管理配置

## 积极方面
- 代码结构清晰
- TypeScript 类型定义完整
- 错误处理基本完善
```

## 文档编写详细说明

Docs Agent 自动生成完整的项目文档：

### 文档类型

**1. 项目文档**
- README.md - 项目主文档
- 快速开始指南
- 贡献指南

**2. API 文档**
- REST API 参考
- 请求/响应示例
- 错误码说明

**3. 架构文档**
- 系统架构图
- 数据库设计
- 技术选型说明

**4. 开发文档**
- 环境搭建
- 代码规范
- 测试指南

**5. 部署文档**
- Docker 部署
- CI/CD 配置
- 生产环境配置

### 文档结构

```
docs/
├── README.md              # 项目主文档
├── getting-started.md     # 快速开始
├── architecture.md        # 架构设计
├── api/                   # API 文档
│   ├── overview.md
│   ├── endpoints.md
│   └── examples.md
├── development/           # 开发文档
│   ├── setup.md
│   ├── coding-standards.md
│   └── testing.md
└── deployment/            # 部署文档
    ├── docker.md
    ├── ci-cd.md
    └── production.md
```

## 最佳实践

### 1. 保持在一个对话窗口

所有步骤必须在同一个 Chat 中进行，不要新建对话。

### 2. 上下文锚点机制

使用 `[[TAG]]` 让 Agent 自动找到需要的上下文：

```markdown
[[PROJECT_GENESIS]]
[内容]
[[PROJECT_GENESIS_END]]
```

### 3. 角色分离

不要让一个 Agent 同时做产品经理和程序员。

### 4. 迭代思维

每个阶段完成后，简单审核确认，再进入下一阶段。

### 5. 命名权威

让 Architect Agent 统一负责所有命名，不要问文件叫什么。

### 6. 质量优先

Code Reviewer Agent 应该在开发完成后立即审查，而不是等到最后。

### 7. 文档同步

Docs Agent 应该在代码变更后及时更新文档。

## 文件结构

```
multi-agent-system/final/
├── .claude/
│   └── skills/
│       └── dev.json                # 主入口 Skill
├── agents/                          # Agent 提示词
│   ├── product.prompt.md           # 产品经理
│   ├── architect.prompt.md         # 架构师
│   ├── backend.prompt.md           # 后端工程师
│   ├── frontend.prompt.md          # 前端工程师
│   ├── devops.prompt.md            # 运维工程师
│   ├── code-reviewer.prompt.md     # 代码审查（v1）
│   └── docs.prompt.md              # 文档工程师（v1）
├── shared/                          # 共享资源
│   ├── README.md                   # 通信协议说明
│   └── communication-protocol.md   # 详细协议规范
├── orchestrator.mjs                 # 可选编排器（v1）
├── project-router.json              # 项目类型路由配置（v2）
├── agent-mcp-config.json            # MCP 工具配置（v2）
├── project.config.example.yaml      # 项目配置示例
├── MCP-GUIDE.md                     # MCP 工具使用指南
└── README.md                        # 本文件
```

## 常见问题

**Q: 如何让系统识别我的项目？**

A: 使用清晰的描述，包含关键词。例如："我想做一个 **web 管理系统**，用于 **用户管理**，需要 **前端界面** 和 **后端 API**"。

**Q: 可以修改技术栈吗？**

A: 可以。系统会推荐技术栈，但你可以在 `project_master.md` 中修改。

**Q: Agents 并行执行会冲突吗？**

A: 不会。系统会根据依赖关系自动决定哪些 Agent 可以并行。

**Q: 如何调试 Agent 输出？**

A: 查看 `project_master.md` 的进度跟踪部分，每个 Agent 的输出都有记录。

**Q: Code Reviewer 会阻止项目完成吗？**

A: 不会。Code Reviewer 提供审查报告，但不会阻止项目推进。你可以根据报告修复问题。

**Q: 必须使用所有 Agents 吗？**

A: 不必。系统会根据项目类型自动选择需要的 Agents。你也可以手动指定。

**Q: shared/ 目录和锚点机制哪个优先？**

A: 锚点机制是主要方式，shared/ 目录作为辅助和持久化存储。两者可以同时使用。

**Q: 如何使用 orchestrator.mjs？**

A: orchestrator.mjs 是可选的备选方案。如果你喜欢传统的配置文件方式，可以使用它。但推荐使用 `/dev` skill。

## 示例

### 示例 1：Web 应用

```
/dev 做一个博客系统，支持文章发布、评论、标签分类
```

**识别结果**: 全栈 Web 应用
**Agents**: product → architect → (backend + frontend + devops) → (code-reviewer + docs)

**执行流程**:
1. Product 分析需求（用户、文章、评论功能）
2. Architect 设计架构（React + Node.js + PostgreSQL）
3. Backend/Frontend/DevOps 并行开发
4. Code Reviewer 审查代码
5. Docs 生成文档

### 示例 2：CLI 工具

```
/dev 写一个命令行工具，可以批量重命名文件
```

**识别结果**: CLI 工具
**Agents**: product → architect → backend → devops → code-reviewer → docs

**执行流程**:
1. Product 分析需求（批量重命名、规则定义）
2. Architect 设计架构（Python + Click）
3. Backend 实现逻辑
4. DevOps 配置打包
5. Code Reviewer 审查代码
6. Docs 生成使用文档

### 示例 3：API 服务

```
/dev 做一个 REST API，提供天气数据查询接口
```

**识别结果**: API 服务
**Agents**: product → architect → backend → devops → code-reviewer → docs

**执行流程**:
1. Product 分析需求（天气查询、数据来源）
2. Architect 设计架构（Express + PostgreSQL + Redis）
3. Backend 实现 API
4. DevOps 配置部署
5. Code Reviewer 审查安全性
6. Docs 生成 API 文档

## 与 v1/v2 版本的区别

| 特性 | v1 | v2 | Final（整合版）|
|------|----|----|----------------|
| 项目识别 | 手动配置文件 | **自动识别** | **自动识别** |
| 启动方式 | `/project` | `/dev [想法]` | `/dev [想法]` |
| 上下文传递 | 文件共享 | **锚点机制** | **锚点 + 文件** |
| 命名管理 | 手动决定 | **Agent 统一** | **Agent 统一** |
| Code Reviewer | ✅ 有 | ❌ 无 | ✅ **有** |
| Docs Agent | ✅ 独立 | ✅ 独立 | ✅ **独立** |
| 详细 Prompt | 简洁版 | **详细版** | **详细版** |
| MCP 工具 | ❌ 无 | ✅ **集成** | ✅ **集成** |
| Orchestrator | ✅ 有 | ❌ 无 | ✅ **可选** |

## 贡献

欢迎提交 Issue 和 Pull Request！

## License

MIT
