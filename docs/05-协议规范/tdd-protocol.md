# TDD Protocol（测试驱动开发协议）

> **版本**: v1.0
> **生效日期**: 2025-01-18
> **适用范围**: 所有涉及代码实现的Agent

---

## 📝 文档时间戳规则

**所有创建的文档必须包含精确到秒的时间戳。**

**时间戳格式**：
```markdown
> **创建时间**: YYYY-MM-DD HH:MM:SS
```

**示例**：
- ✅ 正确：`> **创建时间**: 2026-01-28 14:30:55`
- ❌ 错误：`> **日期**: 2026-01-28`
- ❌ 错误：`> **创建时间**: 2026-01-28`

**原因**：只有日期无法分辨文档修改的先后顺序，精确到秒可以明确版本顺序。

---

## 🎯 目标

确保代码质量，减少bug，提升可维护性。

```
核心原则：
- 先写测试，再写实现
- 小步前进，频繁验证
- 重构优化，保持测试通过
```

---

## 🔄 RED-GREEN-REFACTOR 循环

### **阶段1: RED - 编写失败的测试**

**目标**: 明确需求，定义接口

**操作**:
1. 在实现功能前，先写测试用例
2. 运行测试，确认失败（红色）
3. 提交失败的测试代码

**示例**:

```javascript
// 测试用户注册
describe('User Registration', () => {
  it('should reject duplicate email', async () => {
    // Arrange
    const userService = new UserService();
    await userService.register('test@example.com', 'password123');

    // Act & Assert
    await expect(
      userService.register('test@example.com', 'password456')
    ).rejects.toThrow('Email already exists');
  });
});
```

**检查点**:
- ✅ 测试代码已编写
- ✅ 运行测试，确认失败
- ✅ 失败原因明确（非语法错误）

---

### **阶段2: GREEN - 实现最小可行代码**

**目标**: 通过测试，不做过度设计

**操作**:
1. 编写**刚好能通过测试**的最小代码
2. 运行测试，确认通过（绿色）
3. 提交实现代码

**示例**:

```javascript
// 最小实现（不考虑错误处理、边界情况）
class UserService {
  async register(email, password) {
    if (this.emails?.has(email)) {
      throw new Error('Email already exists');
    }
    this.emails = this.emails || new Set();
    this.emails.add(email);
    return { email };
  }
}
```

**检查点**:
- ✅ 测试通过
- ✅ 代码简单直接
- ✅ 没有过度设计

---

### **阶段3: REFACTOR - 重构优化**

**目标**: 优化代码质量，保持测试通过

**操作**:
1. 改进代码结构、性能、可读性
2. **确保测试依然通过**
3. 提交重构代码

**示例**:

```javascript
// 重构后：使用数据库、密码哈希
class UserService {
  constructor(database) {
    this.db = database;
  }

  async register(email, password) {
    // 检查重复
    const existing = await this.db.users.findOne({ email });
    if (existing) {
      throw new DuplicateEmailError(email);
    }

    // 密码哈希
    const hashedPassword = await bcrypt.hash(password, 10);

    // 保存用户
    const user = await this.db.users.create({
      email,
      password: hashedPassword,
      createdAt: new Date()
    });

    return { id: user.id, email: user.email };
  }
}
```

**检查点**:
- ✅ 测试依然通过
- ✅ 代码质量提升
- ✅ 没有引入新功能

---

## 🎯 TDD集成到工作流

### **在哪个阶段应用TDD？**

```
阶段3a（后端开发）
└─ backend-agent: 必须应用TDD

阶段3b（前端开发）
└─ frontend-agent: 建议应用TDD（组件测试）

阶段3c（集成调试）
└─ integration-agent: 应用TDD（集成测试）

阶段4（测试验证）
└─ qa-agent: 验证TDD覆盖率
```

---

### **backend-agent的TDD工作流**

**1. API端点开发**

```javascript
// RED: 先写测试
describe('POST /api/users', () => {
  it('should create user with valid data', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({ email: 'test@example.com', password: 'secure123' });

    expect(response.status).toBe(201);
    expect(response.body).toHaveProperty('id');
    expect(response.body.email).toBe('test@example.com');
  });
});

// GREEN: 最小实现
app.post('/api/users', async (req, res) => {
  const { email, password } = req.body;
  const user = await db.users.create({ email, password });
  res.status(201).json({ id: user.id, email: user.email });
});

// REFACTOR: 添加验证
app.post('/api/users', async (req, res) => {
  const { email, password } = req.body;

  // 验证
  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password required' });
  }

  const hashedPassword = await bcrypt.hash(password, 10);
  const user = await db.users.create({ email, password: hashedPassword });

  res.status(201).json({ id: user.id, email: user.email });
});
```

**2. 数据库模型**

```javascript
// RED: 测试模型
describe('User Model', () => {
  it('should hash password before save', async () => {
    const user = new User({ email: 'test@example.com', password: 'plain' });
    await user.save();

    expect(user.password).not.toBe('plain');
    expect(user.password.length).toBe(60); // bcrypt length
  });
});

// GREEN: 最小实现
userSchema.pre('save', async function() {
  this.password = await bcrypt.hash(this.password, 10);
});

// REFACTOR: 添加条件判断
userSchema.pre('save', async function() {
  if (!this.isModified('password')) return;

  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
});
```

---

### **frontend-agent的TDD工作流**

**1. React组件测试**

```jsx
// RED: 先写测试
import { render, screen, fireEvent } from '@testing-library/react';

describe('LoginForm', () => {
  it('should show error on empty submit', () => {
    render(<LoginForm />);
    fireEvent.click(screen.getByText('Login'));
    expect(screen.getByText('Email is required')).toBeInTheDocument();
  });
});

// GREEN: 最小实现
function LoginForm() {
  const [error, setError] = useState('');
  const handleSubmit = () => {
    setError('Email is required');
  };
  return <button onClick={handleSubmit}>Login</button>;
}

// REFACTOR: 完整表单
function LoginForm() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!email) {
      setError('Email is required');
      return;
    }
    // 实际登录逻辑...
  };

  return (
    <form onSubmit={handleSubmit}>
      <input data-testid="email" value={email} onChange={(e) => setEmail(e.target.value)} />
      {error && <div className="error">{error}</div>}
      <button>Login</button>
    </form>
  );
}
```

---

## 📊 测试覆盖率目标

| 测试类型 | 最低覆盖率 | 推荐覆盖率 |
|---------|-----------|-----------|
| **单元测试** | 80% | 90%+ |
| **集成测试** | 60% | 75%+ |
| **端到端测试** | 关键流程 | 主要流程 |

---

## 🛡️ 常见陷阱

### **陷阱1: 写完代码再补测试**

❌ **错误**:
```
1. 写完整功能
2. 写测试覆盖所有情况
3. 发现bug，再改代码
```

✅ **正确**:
```
1. 写一个测试用例
2. 写最小实现
3. 重构优化
4. 重复
```

---

### **陷阱2: 测试实现细节**

❌ **错误**:
```javascript
// 测试私有方法
it('should call _validateEmail', () => {
  spyOn(userService, '_validateEmail');
  // ...
});
```

✅ **正确**:
```javascript
// 测试公开行为
it('should reject invalid email', async () => {
  await expect(
    userService.register('invalid-email', 'pass')
  ).rejects.toThrow('Invalid email');
});
```

---

### **陷阱3: 过度Mock**

❌ **错误**:
```javascript
// Mock所有依赖
jest.mock('./database');
jest.mock('./email-service');
jest.mock('./logger');
// 测试变成测试Mock，不是测试逻辑
```

✅ **正确**:
```javascript
// 只Mock外部依赖（数据库、API）
jest.mock('./database');
// 保持内部逻辑真实
```

---

## ✅ TDD检查清单

### **写测试时（RED阶段）**

```
□ 测试用例覆盖核心功能
□ 测试用例覆盖边界情况
□ 测试用例覆盖错误处理
□ 运行测试，确认失败
□ 失败原因明确
```

### **写实现时（GREEN阶段）**

```
□ 只写刚好能通过测试的代码
□ 不做过度设计
□ 不考虑性能优化
□ 运行测试，确认通过
```

### **重构时（REFACTOR阶段）**

```
□ 改进代码结构
□ 提取重复代码
□ 优化性能
□ 确保测试依然通过
□ 没有引入新功能
```

---

## 📚 参考资源

- **测试框架**:
  - JavaScript: Jest, Vitest
  - Python: pytest
  - Go: testing package

- **Mock工具**:
  - JavaScript: jest.mock, sinon
  - Python: unittest.mock
  - Go: testify/mock

---

**文档状态**: 活跃
**维护者**: Claude Code AI System
**下次更新**: 根据实际使用反馈优化
