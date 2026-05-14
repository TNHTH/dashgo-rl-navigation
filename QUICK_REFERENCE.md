# 🚀 Claude Code 智能Agent工作流系统 - 快速参考

> **版本**: v4.0 Pro
> **创建时间**: 2025-01-18
> **更新**: 新增工程化协议（TDD + Debugging + Code Review）

---

## 🎭 8个智能Agent速查表

| Agent | 触发关键词 | 阶段 | 工程化协议 | 主要输出 |
|-------|----------|------|-----------|---------|
| **product-agent** | 需求分析、PRD、用户故事 | 阶段1 | - | requirements.md, user-stories.md |
| **architect-agent** | 架构设计、技术栈、API设计 | 阶段2 | - | tech-stack.md, system-design.md |
| **backend-agent** | 后端开发、API实现、数据库 | 阶段3a | **TDD** | backend-code/, test-suites/, api-implementation.md |
| **frontend-agent** | 前端开发、UI组件、界面 | 阶段3b | TDD（建议） | frontend-code/, component-tests/, component-catalog.md |
| **integration-agent** | 集成、联调、docker-compose | 阶段3c | **Code Review** | docker-compose.yml, integration-report.md, code-review-report.md |
| **qa-agent** | 测试、验证、QA | 阶段4 | - | test-plan.md, test-report.md |
| **devops-agent** | 部署、上线、Dockerfile | 阶段5 | - | Dockerfile, deployment-guide.md |
| **red-team-agent** | 安全审计、漏洞扫描 | 安全审计 | - | security-report.md, vulnerability-list.md |

**注**: `**加粗**`表示强制应用，不加粗表示建议应用

---

## 💾 备份与回滚命令

### 创建备份
```bash
./tools/maintenance/backup-phase.sh <阶段> <Agent>
```
**示例**:
```bash
./tools/maintenance/backup-phase.sh phase1 product-agent
./tools/maintenance/backup-phase.sh phase2 architect-agent
```

### 查看备份
```bash
./tools/maintenance/list-backups.sh
```

### 回滚命令

**本地快照回滚**（优先）:
```bash
./tools/maintenance/rollback.sh .backups/phase-X-TIMESTAMP
```

**Git回滚**（如果Git可用）:
```bash
./tools/maintenance/rollback.sh git 1          # 回滚1步
./tools/maintenance/rollback.sh git <hash>     # 回滚到指定commit
```

---

## 🔧 工程化协议速查

### 1️⃣ TDD Protocol（测试驱动开发）

**文档**: `docs/05-协议规范/tdd-protocol.md`
**适用**: backend-agent（强制）、frontend-agent（建议）

**RED-GREEN-REFACTOR循环**:
```
RED    → 写失败测试（明确需求）
GREEN  → 最小实现（通过测试）
REFACTOR → 重构优化（保持通过）
```

**测试覆盖率要求**:
- 后端：≥80%
- 前端：≥60%

**快速示例**:
```javascript
// RED: 先写测试
it('should reject duplicate email', async () => {
  await expect(
    userService.register('test@example.com', 'pass')
  ).rejects.toThrow('Email already exists');
});

// GREEN: 最小实现
async register(email, password) {
  if (this.emails?.has(email)) throw new Error('Email already exists');
  this.emails = this.emails || new Set();
  this.emails.add(email);
}

// REFACTOR: 优化（添加数据库、密码哈希等）
```

---

### 2️⃣ Systematic Debugging（系统化调试）

**文档**: `docs/05-协议规范/debugging-protocol.md`
**适用**: 所有Agent，出现bug时必须

**四步流程**:
```
Reproduce   → 复现问题（写失败测试）
Locate      → 定位根因（堆栈跟踪、日志）
Hypothesize → 提出假设（基于证据）
Verify      → 验证修复（测试通过）
```

**常见Bug模式**:
- 异步竞态条件 → `Promise.all`
- 状态未更新 → `useState`
- 内存泄漏 → `useEffect`清理
- SQL注入 → 参数化查询
- XSS漏洞 → 输出编码

---

### 3️⃣ Two-Stage Code Review（两阶段代码审查）

**文档**: `docs/05-协议规范/code-review-protocol.md`
**适用**: integration-agent（阶段3c完成时必须）

**阶段1：规范符合性**（architect-agent + integration-agent）:
```
□ 功能符合PRD
□ 架构符合设计
□ API符合契约
□ 数据模型符合schema
```

**阶段2：代码质量**（backend-agent + frontend-agent + red-team-agent）:
```
□ 可读性（命名清晰、结构合理）
□ 性能（无N+1查询、有索引）
□ 安全性（输入验证、防注入）
□ 可维护性（单一职责、测试覆盖）
```

**审查结果**:
- ✅ 通过 → 进入qa-agent测试
- ⚠️ 有建议 → 可以合并，但创建优化任务
- ❌ 不通过 → 必须修改，重新审查

---

## 📂 项目目录结构

```
项目根目录/
├── .artifacts/              # 所有Agent产物
│   ├── phase1-product/     # 需求分析
│   ├── phase2-architecture/# 架构设计
│   ├── phase3-backend/     # 后端开发
│   ├── phase3-frontend/    # 前端开发
│   ├── phase3-integration/ # 集成调试
│   ├── phase4-testing/     # 测试验证
│   ├── phase5-deployment/  # 部署上线
│   └── security/          # 安全审计报告
├── .backups/              # 本地快照备份
├── docs/
│   ├── INDEX.md           # Artifact索引（重要！）
│   ├── tdd-protocol.md           # TDD协议
│   ├── debugging-protocol.md     # 系统化调试协议
│   └── code-review-protocol.md   # 两阶段代码审查协议
├── tools/
│   └── maintenance/       # 维护脚本
│       ├── init-project.sh
│       ├── backup-phase.sh
│       ├── rollback.sh
│       └── list-backups.sh
├── .claude/
│   └── instructions.md    # 系统配置（v4.0 Pro）
├── QUICK_REFERENCE.md     # 本文件
└── README.md
```

---

## 🎯 典型工作流

### 启动新项目
```bash
# 1. 初始化项目
./tools/maintenance/init-project.sh "我的项目名"

# 2. 启动Claude Code
claude

# 3. 在Claude中描述需求
我要开发一个[项目描述]
```

### 逐阶段推进
```
你: "需求确认，继续架构设计"
     ↓
Claude: 自动识别architect-agent
     ↓
完成: 自动保存artifacts + 更新INDEX + 创建备份
     ↓
你: "架构确认，开始后端开发"
     ↓
...（依此类推）
```

### Context清理
```bash
# 对话过长时
/clear

# Claude自动读取INDEX.md恢复上下文
继续工作...
```

---

## 🛡️ 安全审计

**自动介入时机**:
- ✅ 阶段2后（架构安全）
- ✅ 阶段3后（代码安全）
- ✅ 阶段5后（部署安全）

**问题分级**:
- 🔴 高危：立即修复
- 🟡 中危：本周修复
- 🟢 低危：延后处理

---

## ⚠️ 紧急操作

### 停止当前任务
```
停止！
```

### 切换Agent
```
切换到[Agent名称]
```

### 回滚到上一阶段
```bash
# 查看备份
./tools/maintenance/list-backups.sh

# 回滚
./tools/maintenance/rollback.sh .backups/phase-X-[最新备份]
```

---

## 📝 Artifact命名规范

- 使用小写字母和连字符
- 格式：`{entity}-{type}.{ext}`
- 示例：`api-contract.md`, `user-auth-flow.md`

---

## 🔧 常见问题

### Q: 如何恢复/clear后的上下文？
**A**: Claude会自动读取`docs/INDEX.md`恢复上下文

### Q: Git不可用怎么办？
**A**: 系统会自动降级到本地快照备份，不影响使用

### Q: 如何跳过安全审计？
**A**: 不推荐，但可以明确说"跳过安全审计继续"

### Q: 如何修改已完成的工作？
**A**: 直接在Claude中说"修改XXX"，会自动识别并更新

---

## 📚 更多信息

- **完整配置**: `.claude/instructions.md`
- **项目索引**: `docs/INDEX.md`
- **项目说明**: `README.md`

---

**最后更新**: 2025-01-18
**维护者**: Claude Code AI Agent System
