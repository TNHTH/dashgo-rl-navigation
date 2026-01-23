# Two-Stage Code Review Protocol（两阶段代码审查协议）

> **版本**: v1.0
> **生效日期**: 2025-01-18
> **适用范围**: 阶段3完成后（集成调试）、qa-agent验证前

---

## 🎯 目标

确保代码符合设计规范且质量达标，防止技术债务累积。

```
核心理念：
- 第一阶段：规范性（做正确的事）
- 第二阶段：质量（正确地做事）
- 双阶段独立评估，避免混淆
```

---

## 📋 两阶段审查流程

### **阶段1: 规范符合性检查**

**审查者**: architect-agent、integration-agent
**审查重点**: 是否符合PRD、技术架构、API契约

#### **检查清单**

**1.1 功能符合性**
- ✅ 实现了PRD中的所有功能点
- ✅ 没有超出需求范围的额外功能（避免镀金）
- ✅ 用户故事被正确实现
- ✅ 边界情况已处理

**1.2 架构符合性**
- ✅ 遵循技术栈选择
- ✅ 分层架构正确（Controller → Service → Repository）
- ✅ 模块划分合理
- ✅ 依赖关系符合设计

**1.3 API契约符合性**
- ✅ 端点路径符合API设计文档
- ✅ 请求/响应格式符合OpenAPI规范
- ✅ HTTP方法使用正确（GET/POST/PUT/DELETE）
- ✅ 状态码使用正确

**1.4 数据模型符合性**
- ✅ 数据库schema符合设计
- ✅ 字段类型、约束正确
- ✅ 关系定义正确（1:1, 1:N, N:M）
- ✅ 索引策略符合设计

#### **审查示例**

```javascript
// PRD要求: "用户可以注册邮箱和密码"
// API设计: POST /api/users { email, password }

// ❌ 不符合规范
// 1. 缺少email字段
app.post('/api/users', (req, res) => {
  const { username, password } = req.body; // ❌ PRD说是email
});

// 2. 端点路径错误
app.post('/api/register', ...); // ❌ API设计说是/api/users

// ✅ 符合规范
app.post('/api/users', async (req, res) => {
  const { email, password } = req.body; // ✅ 字段名正确

  // 验证
  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password required' });
  }

  // 创建用户
  const user = await userService.create({ email, password });
  res.status(201).json({ id: user.id, email: user.email });
});
```

#### **审查结果**

- ✅ **通过**: 进入第二阶段
- ❌ **不通过**: 退回修改，重新审查

---

### **阶段2: 代码质量评估**

**审查者**: backend-agent、frontend-agent、red-team-agent
**审查重点**: 可读性、性能、安全性、可维护性

#### **检查清单**

**2.1 可读性**
- ✅ 命名清晰（变量、函数、类）
- ✅ 代码结构逻辑清晰
- ✅ 注释恰当（不过度，也不缺失）
- ✅ 遵循代码风格指南（ESLint、Prettier）
- ✅ 没有重复代码（DRY原则）

**2.2 性能**
- ✅ 没有明显的性能问题（N+1查询、大循环）
- ✅ 数据库查询优化（索引、分页）
- ✅ 缓存策略合理
- ✅ 异步操作正确（async/await）
- ✅ 资源释放（连接、内存）

**2.3 安全性**
- ✅ 输入验证（所有用户输入）
- ✅ 输出编码（防止XSS）
- ✅ SQL注入防护（参数化查询）
- ✅ 认证授权正确
- ✅ 敏感数据不泄露（密码、token）
- ✅ 依赖包无已知漏洞

**2.4 可维护性**
- ✅ 单一职责（函数/类只做一件事）
- ✅ 低耦合（模块间依赖最小）
- ✅ 高内聚（相关功能聚集）
- ✅ 错误处理完善
- ✅ 日志记录恰当
- ✅ 测试覆盖率达标

#### **审查示例**

```javascript
// ❌ 代码质量问题

// 问题1: 命名不清晰
const d = await getUserData(u); // d是什么？

// 问题2: 安全漏洞
const query = `SELECT * FROM users WHERE id = ${userId}`; // SQL注入
const user = await db.query(query);

// 问题3: 性能问题
for (const user of users) {
  const posts = await db.query('SELECT * FROM posts WHERE userId = ?', user.id); // N+1查询
  user.posts = posts;
}

// ✅ 修复后

// 修复1: 清晰命名
const userData = await getUserData(userId);

// 修复2: 参数化查询
const user = await db.query(
  'SELECT * FROM users WHERE id = ?',
  [userId]
);

// 修复3: 批量查询
const userIds = users.map(u => u.id);
const posts = await db.query(
  'SELECT * FROM posts WHERE userId IN (?)',
  [userIds]
);
const postsByUser = groupBy(posts, 'userId');
users.forEach(user => {
  user.posts = postsByUser[user.id] || [];
});
```

#### **审查结果**

- ✅ **通过**: 进入qa-agent测试
- ⚠️ **有建议**: 可以合并，但建议优化（非阻塞）
- ❌ **不通过**: 必须修改，重新审查

---

## 🎯 审查触发时机

```
阶段3a完成（后端开发）
└─ backend-agent自查
└─ integration-agent规范审查
└─ architect-agent最终确认

阶段3b完成（前端开发）
└─ frontend-agent自查
└─ integration-agent规范审查
└─ architect-agent最终确认

阶段3c完成（集成调试）
└─ integration-agent自查
└─ 完整的两阶段审查
└─ 进入qa-agent测试
```

---

## 📊 审查报告模板

### **阶段1审查报告**

```markdown
# 代码审查报告 - 阶段1：规范性

**审查对象**: 后端用户管理模块
**审查者**: architect-agent
**审查时间**: 2025-01-18

## 功能符合性
- ✅ 实现了PRD中的所有功能点
- ✅ 用户注册、登录、密码重置
- ⚠️ 缺少邮箱验证功能（PRD中有提到）

## 架构符合性
- ✅ 遵循分层架构（Controller → Service → Repository）
- ✅ 模块划分合理
- ❌ UserController直接调用数据库（应通过Service层）

## API契约符合性
- ✅ 端点路径符合API设计文档
- ✅ 请求/响应格式正确
- ❌ POST /api/users 返回201，但设计文档说是200

## 数据模型符合性
- ✅ 数据库schema符合设计
- ✅ 字段类型正确

## 审查结论
❌ **不通过**，需修改：
1. 添加邮箱验证功能
2. UserController改为调用UserService
3. 修改POST /api/users返回状态码为200
```

---

### **阶段2审查报告**

```markdown
# 代码审查报告 - 阶段2：代码质量

**审查对象**: 后端用户管理模块
**审查者**: backend-agent, red-team-agent
**审查时间**: 2025-01-18

## 可读性
- ✅ 命名清晰
- ✅ 代码结构逻辑清晰
- ⚠️ 部分函数过长（>100行），建议拆分
- ✅ 遵循ESLint规范

## 性能
- ⚠️ 用户列表查询存在N+1问题
- ✅ 使用了分页
- ❌ 缺少数据库索引
- ✅ 异步操作正确

## 安全性
- ✅ 输入验证完善
- ✅ 密码使用bcrypt哈希
- ❌ 缺少CSRF保护
- ⚠️ JWT token未设置过期时间

## 可维护性
- ✅ 单一职责
- ✅ 错误处理完善
- ⚠️ 缺少单元测试（覆盖率30%）
- ✅ 日志记录恰当

## 审查结论
⚠️ **有建议**，建议优化：
- 修复N+1查询问题
- 添加数据库索引
- 添加CSRF保护
- 设置JWT过期时间
- 提高测试覆盖率到80%

（非阻塞，可以合并，但建议在本迭代完成）
```

---

## 🛡️ 审查最佳实践

### **DO ✅**

- **先自动化，后人工**: 先运行linter、测试，再人工审查
- **建设性反馈**: 指出问题的同时提供解决方案
- **解释原因**: 不仅是"这样做"，更是"为什么"
- **关注重要问题**: 优先处理安全、性能、架构问题
- **认可好代码**: 看到优秀的代码应该明确赞赏

### **DON'T ❌**

- **人身攻击**: 不要说"这代码写的什么垃圾"，要说"这里可以改进"
- **微观管理**: 不要纠结缩进、命名风格（让linter处理）
- **一次性改太多**: 不要提出50条修改意见，分优先级
- **拖延审查**: 不要让代码等了好几天才审查
- **假设意图**: 如果不理解，先问"为什么这样做"，不要直接批评

---

## 📋 完整审查流程

```
代码提交
    ↓
┌─────────────────────────────────────┐
│ 阶段1: 规范符合性检查              │
├─────────────────────────────────────┤
│ 1. 功能符合性                      │
│ 2. 架构符合性                      │
│ 3. API契约符合性                   │
│ 4. 数据模型符合性                  │
└─────────────────────────────────────┘
    ↓
  通过？
    ├─ 是 → 进入阶段2
    └─ 否 → 退回修改
            ↓
        重新提交

┌─────────────────────────────────────┐
│ 阶段2: 代码质量评估                │
├─────────────────────────────────────┤
│ 1. 可读性                          │
│ 2. 性能                            │
│ 3. 安全性                          │
│ 4. 可维护性                        │
└─────────────────────────────────────┘
    ↓
  通过？
    ├─ 是 → 进入qa-agent测试
    └─ 有建议 → 可以合并，但建议优化
            ↓
        创建优化任务
```

---

## 🔗 与其他协议的集成

```
Code Review + TDD:
└─ Review检查测试覆盖率
└─ Review验证测试质量

Code Review + Systematic Debugging:
└─ Review发现bug → 触发Debugging
└─ Debugging完成 → 重新Review

Code Review + Red Team:
└─ Red Team参与阶段2安全性审查
└─ 发现严重安全问题 → 阻塞合并
```

---

## ✅ 审查检查清单（快速版）

### **阶段1: 规范性**

```
□ 功能符合PRD
□ 架构符合设计
□ API符合契约
□ 数据模型符合schema
```

### **阶段2: 质量**

```
□ 代码可读
□ 性能合理
□ 安全无漏洞
□ 可维护性好
□ 测试覆盖充分
```

---

## 📚 参考资源

- **代码审查工具**:
  - GitHub Pull Requests
  - GitLab Merge Requests
  - Bitbucket Code Review

- **自动化工具**:
  - Linter: ESLint, Pylint, golangci-lint
  - Formatter: Prettier, Black, gofmt
  - Security: Snyk, OWASP Dependency Check

---

**文档状态**: 活跃
**维护者**: Claude Code AI System
**下次更新**: 根据审查实践补充案例
