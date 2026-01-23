# Shared Resources

这个目录包含在多个 Agent 之间共享的资源和通信契约。

## 目录结构

```
shared/
├── communication-protocol.md    # Agent 通信协议规范
├── types/                       # 共享类型定义
│   └── api.ts                   # API 请求/响应类型
├── schemas/                     # 数据模型 Schema
│   └── user.json                # 用户数据模型
├── contracts/                   # API 契约
│   └── openapi.yaml             # OpenAPI 规范
└── config/                      # 共享配置
    └── env.template.json        # 环境变量模板
```

## 通信方式

Multi-Agent 系统支持三种 Agent 间通信方式：

### 1. 文件共享（推荐）

Agent 通过读写共享目录中的文件进行通信：

- **Backend Agent** 写入 `shared/contracts/backend-api.yaml`
- **Frontend Agent** 读取并生成客户端代码
- **Docs Agent** 收集所有文件生成文档

### 2. 锚点机制（v2 特性）

使用 `[[TAG]]` 格式在对话历史中标记上下文：

```markdown
[[PROJECT_GENESIS]]
项目宪法内容
[[PROJECT_GENESIS_END]]

[[API_CONTRACT]]
API 定义
[[API_CONTRACT_END]]
```

Agent 扫描对话历史，自动找到对应的上下文。

### 3. 消息传递

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

## 依赖关系

在 `project.config.yaml` 中定义 Agent 依赖：

```yaml
agents:
  backend:
    dependencies: []
    output:
      - "shared/contracts/*.yaml"

  frontend:
    dependencies: [backend]
    input:
      - "shared/contracts/*.yaml"
    output:
      - "src/api/*.ts"
```

## 最佳实践

1. **明确定义接口**: 在 `shared/contracts/` 中明确定义 API 契约
2. **使用类型安全**: 通过 TypeScript 类型确保前后端一致
3. **版本化**: 对共享的 Schema 进行版本控制
4. **文档先行**: Backend 先输出 API 文档，Frontend 再开始实现
5. **增量更新**: Agent 可以更新共享文件，下游 Agent 读取最新版本

## 错误处理

如果共享文件读取失败，Agent 应该：

1. 记录错误日志
2. 尝试使用默认值或 Mock 数据
3. 在输出中标记缺失的依赖
4. 继续执行而非阻塞整个流程

## 相关文档

- [通信协议详细说明](communication-protocol.md)
- [项目配置示例](../project.config.example.yaml)
- [Agent Prompt 模板](../agents/)
