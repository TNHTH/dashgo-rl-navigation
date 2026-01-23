# Agent MCP 工具配置指南

本文档详细说明了每个 Agent 应该使用哪些 MCP (Model Context Protocol) 服务器。

## MCP 服务器概览

| MCP 服务器 | 主要功能 | 适用场景 |
|-----------|---------|---------|
| **Context7** | 查询最新库文档和代码示例 | 获取框架 API、最佳实践 |
| **GitHub** | 代码搜索和仓库操作 | 查找实际代码示例 |
| **Firecrawl** | 网页搜索和内容抓取 | 市场调研、竞品分析 |
| **Chrome DevTools** | 浏览器自动化和调试 | 前端调试和测试 |

## Agent → MCP 映射

### 1. Product Manager (产品经理)

**主要工具**: Firecrawl, Context7
**辅助工具**: GitHub

| MCP 工具 | 用途 | 示例 |
|---------|------|------|
| **Firecrawl** | 竞品分析、市场调研 | 搜索 "项目管理工具 SaaS 竞品" |
| **Context7** | 产品设计方法 | 查询 "MVP 功能规划最佳实践" |
| **GitHub** | 开源案例 | 搜索类似项目 README |

**工作流**:
```
Firecrawl (市场信息) → Context7 (方法) → 生成 PRD
```

---

### 2. Architect (架构师)

**主要工具**: Context7, GitHub
**辅助工具**: Firecrawl

| MCP 工具 | 用途 | 推荐查询库 |
|---------|------|-----------|
| **Context7** | 技术栈文档 | `/facebook/react`, `/nodejs/node`, `/prisma/prisma` |
| **GitHub** | 架构案例 | 搜索 "React Node.js starter stars:>1000" |
| **Firecrawl** | 技术博客 | 查询架构设计实战经验 |

**工作流**:
```
Context7 (官方文档) → GitHub (实际案例) → 设计架构
```

---

### 3. Backend Engineer (后端工程师)

**主要工具**: Context7, GitHub

| MCP 工具 | 用途 | 推荐查询库 |
|---------|------|-----------|
| **Context7** | 框架 API 文档 | `/expressjs/express`, `/nestjs/nest`, `/prisma/prisma` |
| **GitHub** | 代码示例 | 搜索 "Express JWT authentication" |

**查询示例**:
```
Context7: "Prisma 如何定义一对多关系?"
GitHub: "REST API best practices boilerplate"
```

**工作流**:
```
Context7 (框架文档) → GitHub (代码示例) → 编写后端
```

---

### 4. Frontend Engineer (前端工程师)

**主要工具**: Context7, Chrome DevTools
**辅助工具**: GitHub

| MCP 工具 | 用途 | 推荐查询库 |
|---------|------|-----------|
| **Context7** | 框架 API | `/facebook/react`, `/tanstack/query`, `tailwindlabs/tailwindcss` |
| **Chrome DevTools** | 调试验证 | 打开页面、截图、检查请求 |
| **GitHub** | 组件示例 | 搜索 "React table component stars:>1000" |

**使用场景**:
```
Context7: "React Hooks 最佳实践"
Chrome DevTools: 打开 http://localhost:3000 并截图
GitHub: "Tailwind CSS dashboard template"
```

**工作流**:
```
Context7 (文档) → 编写代码 → Chrome DevTools (调试) → GitHub (解决方案)
```

---

### 5. DevOps Engineer (运维工程师)

**主要工具**: Context7, GitHub

| MCP 工具 | 用途 | 推荐查询文档 |
|---------|------|-------------|
| **Context7** | 工具文档 | Docker、Kubernetes、GitHub Actions |
| **GitHub** | 配置示例 | 搜索 "Dockerfile multi-stage" |

**查询示例**:
```
Context7: "Docker 多阶段构建最佳实践"
GitHub: "docker-compose.yml production example"
```

**工作流**:
```
Context7 (工具文档) → GitHub (配置示例) → 编写部署配置
```

---

### 6. Documentation Engineer (文档工程师)

**主要工具**: Context7
**辅助工具**: GitHub

| MCP 工具 | 用途 |
|---------|------|
| **Context7** | 文档规范、写作指南 |
| **GitHub** | 优秀文档案例 |

**查询示例**:
```
Context7: "如何编写优秀的 README?"
GitHub: 搜索 star 数高的 Node.js 项目 README
```

---

## MCP 使用最佳实践

### Context7

1. **先解析库 ID**
   ```
   先用: resolve-library-id("react")
   再用: query-docs("/facebook/react", "Hooks 示例")
   ```

2. **最多 3 次查询**
   - 避免重复查询相同内容
   - 优先获取带代码示例的文档

3. **推荐库 ID 格式**
   - GitHub: `/org/repo` 或 `/org/repo/version`
   - NPM: `package-name`

### GitHub

1. **筛选高质量项目**
   ```
   添加: stars:>1000
   添加: pushed:>2024-01-01
   ```

2. **搜索配置文件**
   ```
   搜索: "Dockerfile"
   搜索: ".github/workflows/"
   ```

3. **查看 examples 目录**
   - 优先查看 `/examples` 或 `/samples`
   - 参考知名项目的配置

### Firecrawl

1. **先搜索后抓取**
   ```
   第一步: firecrawl_search("项目管理工具")
   第二步: firecrawl_scrape([找到的页面])
   ```

2. **指定数据源**
   ```
   sources: [{"type": "web"}]
   ```

### Chrome DevTools

1. **仅用于测试验证**
   - 不用于开发代码
   - 主要用于本地调试

2. **典型流程**
   ```
   1. 打开页面: navigate_page("http://localhost:3000")
   2. 截图验证: take_screenshot()
   3. 检查请求: list_network_requests()
   ```

## 优先级规则

### 选择 MCP 的优先级

```
1. 查询文档 → Context7
2. 查找代码 → GitHub
3. 市场调研 → Firecrawl
4. 调试前端 → Chrome DevTools
```

### 每个场景的最佳 MCP

| 场景 | 首选 MCP | 备选 MCP |
|------|---------|---------|
| 技术文档 | Context7 | - |
| 代码示例 | GitHub | Context7 |
| 竞品分析 | Firecrawl | - |
| 前端调试 | Chrome DevTools | - |
| 配置文件 | GitHub | Context7 |
| 最佳实践 | Context7 | GitHub |

## 文件位置

- **主配置**: `agent-mcp-config.json`
- **Agent 提示词**: `agents/*.prompt.md` (每个文件包含对应的 MCP 使用说明)
- **项目路由**: `project-router.json`
- **主 Skill**: `.claude/skills/dev.json`

## 快速参考

```json
{
  "product": ["firecrawl", "context7"],
  "architect": ["context7", "github"],
  "backend": ["context7", "github"],
  "frontend": ["context7", "chrome-devtools"],
  "devops": ["context7", "github"],
  "docs": ["context7", "github"]
}
```
