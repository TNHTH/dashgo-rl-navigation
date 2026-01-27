# Layer 6: Skill评估与优化协议

> **按需加载**: 本文件是dialogue-optimizer skill的扩展协议，仅在评估Skill时加载
> **依赖**: 需要先加载SKILL.md (Layer 1-4)
> **版本**: V5.0 Full-Ecosystem

---

### 触发条件

**自动触发**（满足任一）：
1. 对话中使用了任何Skill（用户调用`/skillname`）
2. 用户说"评估skill"、"分析skill"、"优化skill"
3. 检测到Skill输出质量问题
4. 用户询问"哪个skill好用"、"skill推荐"
5. 发现Skill功能缺失或不完善

### Skill性能评估

#### 1. 触发与调用检测
```yaml
检查项:
  - Skill名称是否清晰？（名称简洁且描述功能）
  - 调用方式是否正确？（/skillname vs Task工具）
  - 是否存在误用？（应该用Agent还是Skill？）
  - 触发关键词是否完整？

示例：
  用户需求：心理分析，期望自动触发
  错误：创建analyze.json（Skill）→ 只能手动/analyze
  正确：创建psychological-counselor.prompt.md（Agent）→ AI自动调用
  评估：违反DR-008，需要重新设计
```

#### 2. 功能完整性评估
```yaml
评估维度:
  - 核心功能：是否覆盖主要使用场景？
  - 边界处理：是否处理异常情况？
  - 可扩展性：是否易于添加新功能？
  - 文档完整性：SKILL.md是否清晰？

评分标准:
  - 95分：功能完整 + 文档清晰 + 易扩展
  - 85分：基本功能完整但文档不足
  - 75分：功能缺失或文档混乱
  - <60分：无法正常使用或严重缺陷
```

#### 3. 输出质量评估
```yaml
评估维度:
  - 准确性：输出是否满足用户需求？
  - 完整性：是否遗漏关键信息？
  - 可读性：输出格式是否清晰？
  - 可执行性：建议是否具体可操作？

常见问题:
  - ❌ 输出泛化内容（"总的来说"、"值得注意的是"）
  - ❌ 只给理论框架，缺少具体案例
  - ❌ 输出格式混乱（列表/表格/段落混用）
  - ✅ 要点列表 + 代码示例 + 具体建议
```

#### 4. Skill vs Agent判断
```yaml
判断标准（参考DR-008）:

**应该创建Skill的情况**:
  - 用户手动触发（/skillname）
  - 功能明确且不需要主动判断
  - 工具类功能（格式化、转换、生成）
  - 示例：brainstorming、kaizen、file-organizer

**应该创建Agent的情况**:
  - AI需要主动判断何时使用
  - 需要自动扫描数据/识别模式
  - 需要自主研究和分析
  - 示例：Code-Reviewer、Architect、Product

**常见误用**:
  - ❌ 用户说"自动心理分析" → 创建Skill（不会自动触发）
  - ✅ 用户说"自动心理分析" → 创建Agent（AI主动调用）
```

#### 5. Token消耗与性价比评估
```yaml
评估维度:
  - 输入Token: 用户输入 + SKILL.md加载
  - 输出Token: AI生成的输出
  - 总消耗: 输入 + 输出
  - 功能价值: 综合评分（0-100分）
  - 性价比: 功能价值 / Token消耗 × 100

Token估算方法:
  1. 输入Token: 用户输入字符数 / 4 + SKILL.md大小
  2. 输出Token: 输出字符数 / 4
  3. 总消耗: 输入 + 输出

评分标准:
  - 🟢 高性价比: > 3.0（Skill通常消耗低，性价比应该高）
  - 🟡 中性价比: 1.5-3.0
  - 🔴 低性价比: < 1.5（需要优化输出或精简文档）

示例：
  Skill: brainstorming
  输入: 1K tokens (用户输入 + SKILL.md)
  输出: 2K tokens (方案列表)
  总消耗: 3K tokens
  功能价值: 81分
  性价比: 81 / 3000 × 100 = 2.70
  评级: 🟢 高性价比

  对比:
  - dialogue-optimizer: 5K tokens, 94分, 性价比18.8 ✅
  - brainstorming: 3K tokens, 81分, 性比2.70 ✅
  - kaizen: 4K tokens, 75分, 性价比1.88 🟡
```

### Skill优化建议

#### 1. 功能缺失分析
```yaml
检查清单:
  □ 是否有核心功能缺失？
  □ 是否处理了异常情况？
  □ 是否有清晰的执行流程？
  □ 是否有输出格式规范？

示例优化：
  Skill: brainstorming
  问题：缺少输出模板
  优化：
    - 添加结构化输出模板
    - 添加优先级排序机制
    - 添加可行性评估
```

#### 2. 文档改进建议
```yaml
SKILL.md质量检查:
  □ name是否清晰？（简洁且描述功能）
  □ description是否准确？（一句话说明用途）
  □ 是否有使用示例？
  □ 是否有参数说明？
  □ 是否有Layer结构？（可选，复杂skill）

改进示例：
  Before: "description: 头脑风暴"
  After: "description: 结构化头脑风暴，通过强制视角切换和逆向思维突破思维定式，生成创新方案"

  Before: 无使用示例
  After:
    ```markdown
    ## 使用示例
    用户：/brainstorming 如何提升学习效率？
    输出：
      1. 逆向思维：为什么学习效率低？
      2. 强制视角：如果是AI教练会如何建议？
      3. SCAMPER：如何替换/合并学习方式？
    ```
```

#### 3. 调用方式优化
```yaml
调用方式对比:

**Skill调用**（手动触发）:
  /brainstorming
  /kaizen
  /dialogue-optimizer

**Agent调用**（AI主动）:
  Task({
    subagent_type: "general-purpose",
    prompt: "调用code-reviewer agent分析这段代码"
  })

优化建议:
  - Skill：简化调用方式（短名称更好）
  - Agent：添加触发条件（CLAUDE.md Trigger配置）
```

#### 4. 协作优化建议
```yaml
Skill与其他组件的协作:

**Skill + Agent协作**:
  dialogue-optimizer（Skill）→ 评估对话 → 发现问题 → 更新Agent
  brainstorming（Skill）→ 生成方案 → Architect（Agent）→ 实施架构

**Skill + Skill协作**:
  kaizen（持续改进）→ brainstorming（头脑风暴）→ tapestry（执行计划）

优化方向:
  - 定义清晰的输入/输出接口
  - 添加中间数据格式规范
  - 建立调用链路文档
```

### Skill评估报告模板

```markdown
---
## 🛠️ Skill评估报告

### Skill基本信息
- **名称**: brainstorming
- **路径**: .claude/skills/brainstorming/
- **版本**: V1.0
- **评估时间**: 2026-01-23 15:00:00

### 性能评分
- **功能完整性**: 85/100
- **文档质量**: 70/100
- **输出质量**: 90/100
- **可扩展性**: 80/100
- **综合评分**: 81/100

### Token消耗评估
- **输入Token**: 1K（用户输入 + SKILL.md）
- **输出Token**: 2K（方案列表）
- **总消耗**: 3K tokens
- **功能价值**: 81/100
- **性价比**: 2.70（🟢 高）
- **对比**: dialogue-optimizer性价比18.8，kaizen性价比1.88

### 主要问题

#### ⚠️ 功能缺失（1个）
1. **缺少优先级排序**
   - 问题：生成的方案没有优先级标记
   - 影响：用户不知道应该先执行哪个
   - 建议：添加"高/中/低"优先级标记

#### 📝 文档问题（2个）
1. **description不清晰**
   - 当前："description: 头脑风暴"
   - 改进："description: 结构化头脑风暴，通过强制视角切换和逆向思维突破思维定式"

2. **缺少使用示例**
   - 问题：用户不知道如何使用
   - 建议：添加具体示例（输入/输出）

### 优化建议

#### 1. 功能增强
```markdown
添加优先级评估机制：
- 高优先级：低成本高收益（<1小时，收益>30%）
- 中优先级：中等成本中等收益（1-4小时，收益10-30%）
- 低优先级：高成本低收益（>4小时，收益<10%）
```

#### 2. 文档完善
```markdown
## 使用示例
用户：/brainstorming 如何提升学习效率？

输出结构：
### 方案列表（按优先级排序）
1. [🔴 高] 间隔重复法（每天复习，记忆+50%）
2. [🟡 中] 费曼技巧（解释给别人，理解+30%）
3. [🟢 低] 思维导图（整理知识，效率+10%）

### 实施计划
- 本周执行：方案1（间隔重复法）
- 下周执行：方案2（费曼技巧）
- 有时间执行：方案3（思维导图）
```

#### 3. 调用优化
- ✅ 当前：`/brainstorming`（简洁）
- ✅ 建议：保持现有调用方式

#### 4. 协作建议
```yaml
协作链路:
  brainstorming（生成方案）
    → kaizen（持续改进优化）
    → tapestry（生成执行计划）

示例场景：
  用户：我想优化我的工作流程
  流程：
    1. /brainstorming 生成优化方案
    2. /kaizen 评估方案可行性
    3. /tapestry 生成具体执行计划
```

### 关键洞察

1. **结构化输出 > 自由输出**
   - ❌ 自由列出10个方案
   - ✅ 按优先级 + 成本收益 + 实施计划

2. **文档清晰 > 功能强大**
   - ❌ 功能强大但文档混乱
   - ✅ 功能简洁 + 文档清晰

3. **示例驱动 > 理论说明**
   - ❌ 只讲原理（"头脑风暴是一种..."）
   - ✅ 提供示例（输入/输出对比）

### 学习记录
- 完整文档：`.claude-temp/brainstorming-learning_{date}/对话学习记录_{序号}.md`
- 包含：问题分析、优化建议、更新计划

---
```

### 自动更新Skill功能

```yaml
触发条件:
  - 评估评分 < 80分
  - 发现功能缺失（≥2个）
  - 文档质量 < 70分
  - 用户确认更新

自动更新流程:
  1. 读取Skill文件（SKILL.md）
  2. 生成更新计划：
     - 功能增强（添加新Layer/Section）
     - 文档改进（优化description、添加示例）
     - 格式优化（结构调整）
  3. 等待用户确认（DR-016）
  4. 执行更新（Edit工具）
  5. 创建学习记录

示例：
  Skill: brainstorming
  当前评分: 81分
  更新计划:
    - 优化description（一句话说明价值）
    - 添加使用示例（输入/输出）
    - 添加优先级排序机制
    - 添加输出模板
```

### Skill生态系统评估

```yaml
评估目标:
  - 当前skills列表是否完整？
  - 是否有功能重叠？
  - 是否有功能缺失？
  - skills与agents的协作是否清晰？

当前生态系统（2026-01-23）:

**Skills列表**:
  1. dialogue-optimizer - 对话评估与优化（V4.0）
  2. brainstorming - 结构化头脑风暴
  3. kaizen - 持续改进方法论
  4. file-organizer - 智能文件组织器
  5. tapestry - 统一内容提取和行动规划

**Agents列表**:
  1. Product - 需求分析
  2. Architect - 架构设计
  3. Backend - 后端开发
  4. Frontend - 前端开发
  5. Code-Reviewer - 代码审查与优化（V3.0）
  6. Docs - 文档编写
  7. DevOps - 部署运维

**功能覆盖分析**:
  ✅ 对话优化：dialogue-optimizer
  ✅ 创意生成：brainstorming
  ✅ 持续改进：kaizen
  ✅ 文件管理：file-organizer
  ✅ 内容处理：tapestry
  ✅ 心理咨询：psychological-counselor（Agent）

**潜在缺失**:
  - ❌ 测试自动化skill（类似webapp-testing但更通用）
  - ❌ Git工作流skill（commit规范、PR模板等）
  - ❌ 学习总结skill（自动生成学习笔记）

**建议新增**:
  优先级1：Git工作流skill（高频使用）
  优先级2：测试自动化skill（质量保障）
  优先级3：学习总结skill（知识管理）
```

### 常见Skill问题诊断

```yaml
问题1：Skill不触发
  原因：名称错误或调用方式错误
  诊断：
    - 检查目录名：.claude/skills/{skillname}/
    - 检查调用：/skillname（不是/dialogue_optimizer）
    - 检查SKILL.md：name字段是否匹配

问题2：Skill输出混乱
  原因：缺少输出模板
  解决：添加结构化输出模板
  示例：
    ## 输出格式
    ### 方案列表
    1. [优先级] 方案名称（成本/收益）
    ### 实施计划
    - 本周：...
    - 下周：...

问题3：Skill vs Agent混淆
  原因：违反DR-008
  诊断：
    - 需要AI主动判断？ → Agent
    - 用户手动触发？ → Skill
  示例：
    ❌ 用户要求"自动心理分析" → 创建Skill
    ✅ 用户要求"自动心理分析" → 创建Agent

问题4：Skill功能重叠
  原因：缺少规划
  解决：
    - 合并功能相似的skills
    - 或明确各自使用场景
  示例：
    brainstorming（创意生成）
    vs kaizen（持续改进）
    协作：brainstorming生成方案 → kaizen优化方案
```

### Skill评估触发词

```yaml
自动触发Layer 6的关键词:
  - "评估skill"
  - "分析skill"
  - "优化skill"
  - "哪个skill好用"
  - "skill推荐"
  - "skill不工作"
  - "skill输出质量"

检测信号:
  - 用户使用Skill后不满意
  - 多次使用同一Skill但效果不佳
  - 用户询问"有没有更好的skill"
```

### Prohibited Actions (Skill评估相关)

🚫 **NEVER**:
- 未经用户确认直接修改Skill文件（违反DR-016）
- 基于模板批评Skill而不理解使用场景
- 建议将Agent改为Skill（如果需要主动触发）
- 建议将Skill改为Agent（如果只需要手动触发）
- 忽略Skill的特定用途（如kaizen的持续改进哲学）

✅ **ALWAYS**:
- 理解Skill的核心价值和使用场景
- 务实主义：简单够用 > 完美方案
- 文档优先：清晰的SKILL.md > 复杂功能
- 示例驱动：提供具体使用示例
- 协作优化：考虑Skill与Agent/Skill的协作

---

