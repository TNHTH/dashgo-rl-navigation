---
name: kaizen
description: Use when code implementation and refactoring, architecturing or designing systems, process and workflow improvements, error handling and validation. Provides techniques to avoid over-engineering and apply iterative improvements based on Japanese Kaizen philosophy and Lean methodology.
---

# Kaizen: Continuous Improvement

## 核心理念

小改进，持续进行。设计防错。遵循有效模式。只构建需要的。

**核心原则**: 许多小改进胜过一次大改变。在设计时预防错误，而不是事后修补。

---

## 何时使用

**自动应用于**:
- 代码实现和重构
- 架构和设计决策
- 流程和工作流改进
- 错误处理和验证

**哲学**: 通过增量进步和预防获得质量，而非通过大量努力追求完美。

---

## 四大支柱

### 1. Continuous Improvement (Kaizen)

**原则**:
- 增量优于革命
- 总是让代码比你发现时更好
- 迭代优化：先让它工作，然后清晰，最后高效

**实践**:
- 从最简单的可行版本开始
- 一次改进一个方面
- 每次更改后验证
- 不要立即追求完美

---

### 2. Poka-Yoke (防错设计)

**原则**:
- 让错误不可能发生
- 设计优先于安全
- 分层防御

**实践**:
- 使用类型系统捕获错误
- 在边界验证，内部安全
- 早期失败，明确报错
- 使正确路径明显，错误路径困难

---

### 3. Standardized Work (标准化作业)

**原则**:
- 一致性胜过聪明
- 文档与代码共存
- 自动化标准

**实践**:
- 遵循现有代码库模式
- 不要重新发明已解决的问题
- 新模式仅在显著更好时引入
- 使用linter、type check、CI/CD强制标准

---

### 4. Just-In-Time (即时生产)

**原则**:
- YAGNI (You Aren't Gonna Need It)
- 最简单的可行方案
- 基于测量优化

**实践**:
- 只实现当前需求
- 无"以防万一"的功能
- 不过度设计
- 优化前先测量
- 复杂度仅在需要时添加

---

## 快速检查清单

### 实现功能时
```
□ 从最简单版本开始
□ 一次添加一个改进
□ 测试并验证
□ 时间允许则重复
```

### 处理错误时
```
□ 在系统边界验证
□ 使用类型系统预防错误
□ 早期失败，明确报错
□ 分层防御
```

### 设计API时
```
□ 使用类型约束输入
□ 使无效状态不可表示
□ 在边界验证
□ 返回Result<T, E>而非抛出异常
```

### 抽象时
```
□ 等待3+相似案例（Rule of Three）
□ 最简单的可行抽象
□ 宁可重复不要错误抽象
□ 模式明确时重构
```

---

## 红旗警告

**违反持续改进**:
- "我稍后会重构"（从不发生）
- 让代码比你发现时更差
- 大爆炸式重写而非增量

**违反防错设计**:
- "用户应该小心"
- 使用后验证而非使用前
- 可选配置无验证

**违反标准化**:
- "我更喜欢这样做"
- 不检查现有模式
- 忽略项目约定

**违反即时生产**:
- "我们将来可能需要这个"
- 在使用前构建框架
- 未测量就优化

---

## 集成原则

Kaizen技能指导你如何工作。持续应用这些原则：

**实现时**: 小改进，一次一个
**设计时**: 预防错误，而非修补
**审查时**: 遵循模式，而非创新
**优化时**: 测量优先，优化其次

---

## Optional Reading (按需加载)

深入了解四大支柱：
- `.claude/skills/kaizen/references/four-pillars.md` - 四大支柱详解
- `.claude/skills/kaizen/references/examples.md` - Good/Bad代码示例

历史版本归档在 `archive/kaizen-v1.0.md`

---

**Version**: v2.0 (Lean Runtime)
**Last Updated**: 2026-01-17
**Auto-Apply**: Code implementation, architecture, error handling
