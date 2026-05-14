# Systematic Debugging Protocol（系统化调试协议）

> **版本**: v1.0
> **生效日期**: 2025-01-18
> **适用范围**: 所有Agent，特别是backend-agent、integration-agent、qa-agent

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

系统化地发现和修复bug，避免随机尝试和瞎猜。

```
核心原则：
- 先理解问题，再修复
- 最小化复现步骤
- 基于证据的假设
- 验证修复，防止回归
```

---

## 🔄 四步调试流程

### **步骤1: Reproduce - 复现问题**

**目标**: 稳定复现bug，建立可验证的场景

**操作**:

1. **收集问题信息**
   - 错误消息/堆栈跟踪
   - 预期行为 vs 实际行为
   - 发生频率（每次 / 偶尔 / 难以复现）
   - 环境信息（浏览器、OS、版本）

2. **创建最小复现用例**
   - 去除无关步骤
   - 确定触发条件
   - 记录精确的复现步骤

3. **自动化复现**
   - 编写测试用例来复现bug
   - 这个测试现在会失败（RED）

**示例**:

```javascript
// Bug报告: "用户登录后看不到数据"

// 收集信息
// - 错误: "Cannot read property 'map' of undefined"
// - 发生频率: 每次登录后
// - 环境: Chrome 120, 生产环境

// 最小复现
describe('Login Bug', () => {
  it('should load data after login', async () => {
    // 登录
    await login('user@example.com', 'password');

    // 获取数据（失败：data是undefined）
    const data = await fetchData();
    expect(data).toBeDefined(); // ❌ 失败
  });
});
```

**检查点**:
- ✅ 能够稳定复现bug
- ✅ 有失败的测试用例
- ✅ 错误信息清晰

---

### **步骤2: Locate - 定位问题**

**目标**: 找到bug的根源，不是症状

**操作**:

1. **分析堆栈跟踪**
   - 从上往下读（错误发生在哪里）
   - 找到第一行**你的代码**（忽略库代码）

2. **添加调试日志**
   ```javascript
   console.log('Step 1: variable =', variable);
   console.log('Step 2: after API call', response);
   ```

3. **使用调试器**
   - 设置断点
   - 单步执行
   - 检查变量值

4. **二分法定位**
   - 注释掉一半代码
   - 看bug是否还存在
   - 重复缩小范围

**示例**:

```javascript
// 堆栈跟踪
// TypeError: Cannot read property 'map' of undefined
//   at DataView.render (src/components/DataView.js:15)
//   at App (src/App.js:42)

// 定位到 DataView.js:15
function DataView({ data }) {
  return (
    <div>
      {data.items.map(item => ...)} // ❌ data.items是undefined
    </div>
  );
}

// 添加日志
function DataView({ data }) {
  console.log('DataView received:', data); // undefined
  console.log('data.items:', data?.items);  // undefined

  return (
    <div>
      {data.items.map(item => ...)}
    </div>
  );
}
```

**检查点**:
- ✅ 知道bug发生的精确位置
- ✅ 理解为什么会发生
- ✅ 区分症状和根因

---

### **步骤3: Hypothesize - 提出假设**

**目标**: 基于证据提出修复方案，不瞎猜

**操作**:

1. **列出可能原因**
   - 数据格式不匹配
   - 缺少边界检查
   - 异步操作未完成
   - 环境差异

2. **选择最可能的原因**
   - 用证据支持假设
   - 排除低概率原因

3. **设计修复方案**
   - 最小改动
   - 不引入新问题
   - 符合代码规范

**示例**:

```javascript
// 假设1: API返回的数据格式不同
// 证据: console.log显示data是undefined
// 概率: 高

// 假设2: 数据还未加载完成
// 证据: 需要检查loading状态
// 概率: 中

// 假设3: 网络错误
// 证据: 没有看到网络错误
// 概率: 低

// 选择假设1，设计修复
function DataView({ data }) {
  // 添加边界检查
  if (!data || !data.items) {
    return <div>No data available</div>;
  }

  return (
    <div>
      {data.items.map(item => ...)}
    </div>
  );
}
```

**检查点**:
- ✅ 假设有证据支持
- ✅ 考虑了多种可能性
- ✅ 修复方案清晰

---

### **步骤4: Verify - 验证修复**

**目标**: 确认修复有效，且不引入新问题

**操作**:

1. **应用修复**
   - 实施修复方案
   - 保持最小改动

2. **运行复现测试**
   - 失败的测试现在应该通过（GREEN）
   - 如果还是失败，返回步骤2

3. **添加回归测试**
   - 将修复用例加入测试套件
   - 防止未来再次出现

4. **检查副作用**
   - 运行完整测试套件
   - 检查相关功能
   - 确保没有破坏其他地方

**示例**:

```javascript
// 修复后的代码
function DataView({ data }) {
  // 边界检查
  if (!data || !data.items) {
    return <div>No data available</div>;
  }

  return (
    <div>
      {data.items.map(item => <div key={item.id}>{item.name}</div>)}
    </div>
  );
}

// 测试现在通过
it('should load data after login', async () => {
  await login('user@example.com', 'password');
  const data = await fetchData();
  expect(data).toBeDefined(); // ✅ 通过
});

// 添加回归测试
it('should handle missing data gracefully', () => {
  const { container } = render(<DataView data={null} />);
  expect(container).toHaveTextContent('No data available');
});
```

**检查点**:
- ✅ 复现测试通过
- ✅ 添加了回归测试
- ✅ 没有破坏其他功能

---

## 🎯 调试工具箱

### **前端调试**

```
浏览器DevTools:
├─ Console: 查看错误、日志
├─ Network: 检查API请求
├─ Debugger: 断点调试
├─ React DevTools: 组件状态
└─ Redux DevTools: 状态变化

常用命令:
- console.log(variable)
- console.table(array)
- debugger; // 断点
- JSON.stringify(object, null, 2)
```

### **后端调试**

```
Node.js:
├─ console.log() / console.error()
├─ debugger; // 断点（配合--inspect）
├─ Node.js Inspector
└─ debug库

Python:
├─ print() / logging
├─ pdb.set_trace()
├─ ipdb (增强版)
└── logging模块

检查工具:
- 数据库查询日志
- API请求日志
- 错误跟踪（Sentry）
```

---

## 🛡️ 常见Bug模式

### **Bug 1: 异步竞态条件**

```javascript
// ❌ 问题
async function loadData() {
  const data1 = await fetch(url1);
  const data2 = await fetch(url2); // 等待data1完成
  return { data1, data2 };
}

// ✅ 修复
async function loadData() {
  const [data1, data2] = await Promise.all([
    fetch(url1),
    fetch(url2)
  ]);
  return { data1, data2 };
}
```

---

### **Bug 2: 状态未更新**

```javascript
// ❌ 问题
function Counter() {
  let count = 0;
  const increment = () => {
    count++; // 不触发重新渲染
  };
  return <button onClick={increment}>{count}</button>;
}

// ✅ 修复
function Counter() {
  const [count, setCount] = useState(0);
  const increment = () => {
    setCount(count + 1);
  };
  return <button onClick={increment}>{count}</button>;
}
```

---

### **Bug 3: 内存泄漏**

```javascript
// ❌ 问题
useEffect(() => {
  const timer = setInterval(() => {
    console.log('tick');
  }, 1000);
  // 清理函数缺失
}, []);

// ✅ 修复
useEffect(() => {
  const timer = setInterval(() => {
    console.log('tick');
  }, 1000);

  return () => clearInterval(timer); // 清理
}, []);
```

---

## ✅ 调试检查清单

### **步骤1: Reproduce**

```
□ 收集错误信息（堆栈、截图、日志）
□ 确定复现频率（每次/偶尔/难以复现）
□ 创建最小复现用例
□ 编写失败测试（RED）
```

### **步骤2: Locate**

```
□ 分析堆栈跟踪
□ 添加调试日志
□ 使用调试器断点
□ 定位到精确代码行
```

### **步骤3: Hypothesize**

```
□ 列出所有可能原因
□ 选择最可能的原因
□ 设计修复方案
□ 评估副作用
```

### **步骤4: Verify**

```
□ 应用修复
□ 复现测试通过（GREEN）
□ 添加回归测试
□ 运行完整测试套件
□ 检查相关功能
```

---

## 📚 调试最佳实践

### **DO ✅**

- **小步前进**: 每次只改一处，观察效果
- **保持冷静**: 不要盲目改代码
- **记录过程**: 记录尝试过的方案
- **搜索帮助**: Stack Overflow、GitHub Issues
- **代码审查**: 请同事帮忙看代码

### **DON'T ❌**

- **随机修改**: 不要"试试这个，试试那个"
- **忽略警告**: 警告通常是bug的线索
- **过度修复**: 不要一次改太多地方
- **假设环境**: 不要假设"这不可能发生"
- **跳过测试**: 不要因为"看起来没问题"就跳过验证

---

## 🔗 与其他协议的集成

```
Systematic Debugging + TDD:
└─ Reproduce步骤创建失败的测试
└─ Verify步骤确认测试通过

Systematic Debugging + Code Review:
└─ Review发现bug → 触发Debugging
└─ Debugging完成 → 重新Review
```

---

**文档状态**: 活跃
**维护者**: Claude Code AI System
**下次更新**: 根据实际调试案例补充
