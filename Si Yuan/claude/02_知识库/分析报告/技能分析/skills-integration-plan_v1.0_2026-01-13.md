# Skills 完全整合方案

**整合日期**: 2026-01-13
**环境**: Windows (Win32)
**工作目录**: D:\cursor\file

---

## 一、环境分析

### 当前环境
- **操作系统**: Windows
- **路径分隔符**: `\` (反斜杠)
- **Shell**: PowerShell / CMD
- **Claude Skills路径**: 需要确定

### 用户的规则体系
1. **文件组织规则** (`file-organization.md`):
   - 非项目文档 → `D:\cursor\file\Si Yuan\claude\`
   - 临时文件 → `D:\cursor\file\.claude-temp\`
   - 项目文档 → 项目目录

2. **临时文件管理** (`file-organization.md`):
   - 统一存放到 `.claude-temp\`
   - 对话结束后可清理

3. **对话回顾机制** (`dialogue-review-and-auto-update.md`):
   - 每次对话结束自动回顾
   - 识别新规则并更新
   - 记录到对话历史

### 现有Skills
- `multi-agent-system\.claude\skills\mcp-builder\`
- `multi-agent-system\.claude\skills\skill-creator\`
- `multi-agent-system\.claude\skills\webapp-testing\`
- `multi-agent-system\.claude\skills\changelog-generator\`

---

## 二、整合策略

### Skills目录结构

```
D:\cursor\file\
├── .claude\
│   └── skills\                    # 全局skills（新）
│       ├── kaizen\
│       ├── brainstorming\
│       ├── file-organizer\
│       └── tapestry\
├── multi-agent-system\
│   └── .claude\
│       └── skills\                # 项目特定skills（保持）
│           ├── mcp-builder\
│           ├── skill-creator\
│           ├── webapp-testing\
│           └── changelog-generator\
└── Si Yuan\
    └── claude\                    # 技能文档
        ├── skills-usage-guide.md  # 新建：使用指南
        └── awesome-claude-skills分析报告_2026-01-13.md
```

### Windows适配清单

#### 1. 路径转换
```bash
# Unix → Windows
~ → D:\cursor\file
/home/user → D:\cursor\file
/tmp → D:\cursor\file\.claude-temp
/ → \
```

#### 2. 命令替换
```bash
# Bash → PowerShell
mkdir → New-Item -ItemType Directory
rm → Remove-Item
cp → Copy-Item
mv → Move-Item
ls → Get-ChildItem
cat → Get-Content
echo → Write-Output
```

#### 3. 工具可用性
```bash
# 需要检查的Windows工具
- reader (npm) → Windows支持
- trafilatura (pip) → Windows支持
- yt-dlp → Windows支持
- pdftotext → 需要安装poppler-utils for Windows
- curl → Windows 10+内置
```

---

## 三、Skills整合详情

### 3.1 Kaizen Skill

**整合要点**:
1. ✅ 保持原始内容（通用方法论）
2. ✅ 添加Windows代码示例
3. ✅ 集成到现有skills的优化中

**修改点**:
- 添加Windows PowerShell示例
- 保持TypeScript示例（通用）

---

### 3.2 Brainstorming Skill

**整合要点**:
1. ✅ 适配触发条件
2. ✅ 修改文档保存路径
3. ✅ 集成到skill-creator

**修改点**:
```markdown
# 原始
docs/plans/YYYY-MM-DD-<topic>-design.md

# 整合后
D:\cursor\file\Si Yuan\claude\plans\YYYY-MM-DD-<topic>-design.md
```

---

### 3.3 File Organizer Skill

**整合要点**:
1. ✅ **核心整合**: 完全对齐file-organization.md
2. ✅ 修改所有路径为Windows格式
3. ✅ 添加用户的文件组织规则
4. ✅ 集成临时文件管理

**关键修改**:
```powershell
# 添加默认路径配置
$DEFAULT_DOC_PATH = "D:\cursor\file\Si Yuan\claude\"
$TEMP_PATH = "D:\cursor\file\.claude-temp\"

# 文件分类规则
$PROJECT_PATTERNS = @(
    "multi-agent-system",
    ".claude\skills"
)

$DOC_PATTERNS = @(
    "*科普*.md",
    "*分析*.md",
    "*指南*.md",
    "*笔记*.md"
)
```

---

### 3.4 Tapestry Skill

**整合要点**:
1. ✅ Windows命令替换
2. ✅ 路径适配
3. ✅ 保存到Si Yuan\claude\

**修改点**:
```powershell
# 原始
reader "$URL" > temp_article.txt

# Windows
reader "$URL" | Out-File -Encoding UTF8 temp_article.txt

# 保存路径
$CONTENT_DIR = "D:\cursor\file\Si Yuan\claude\extracted-content\"
$PLAN_DIR = "D:\cursor\file\Si Yuan\claude\plans\"
```

---

## 四、现有Skills优化

### 4.1 Skill-Creator优化

**整合Kaizen原则**:
1. 添加"防错设计"章节
2. 强调YAGNI原则
3. 添加Good/Bad示例

**整合Brainstorming原则**:
1. Step 1添加"一次一个问题"
2. 添加多选题示例
3. 分段验证设计

---

### 4.2 MCP-Builder优化

**整合Kaizen原则**:
1. Phase 2添加"防错设计"最佳实践
2. 强调类型系统的重要性
3. 添加错误处理示例

**整合Brainstorming原则**:
1. Phase 1添加结构化问题清单
2. 分段展示实现计划

---

### 4.3 Changelog-Generator优化

**整合Kaizen原则**:
1. 添加持续改进理念
2. 强调小改进的价值
3. 添加迭代改进示例

---

## 五、执行顺序

### 阶段1: 准备（5分钟）
- [x] 创建全局skills目录
- [ ] 创建子目录结构
- [ ] 检查依赖工具

### 阶段2: 添加新Skills（30分钟）
- [ ] kaizen（基础，无代码）
- [ ] brainstorming（需要路径调整）
- [ ] file-organizer（需要大量适配）
- [ ] tapestry（需要Windows命令）

### 阶段3: 优化现有Skills（25分钟）
- [ ] skill-creator（添加原则）
- [ ] mcp-builder（添加防错设计）
- [ ] changelog-generator（添加持续改进）

### 阶段4: 创建使用指南（10分钟）
- [ ] 编写使用指南
- [ ] 创建快速参考

### 阶段5: 验证和文档（10分钟）
- [ ] 更新对话回顾
- [ ] 测试skill触发
- [ ] 创建整合报告

---

## 六、质量标准

### 每个Skill必须满足：

1. ✅ **Windows兼容**: 所有命令在Windows上可用
2. ✅ **规则对齐**: 遵循file-organization.md
3. ✅ **路径正确**: 使用D:\cursor\file\作为根路径
4. ✅ **描述准确**: description清楚说明使用场景
5. ✅ **示例完整**: 包含Windows特定示例
6. ✅ **错误处理**: 考虑Windows特定错误

---

## 七、成功标准

完成后，你应该能够：

1. ✅ 使用"kaizen"获得持续改进建议
2. ✅ 使用"brainstorming"进行结构化设计
3. ✅ 使用"file-organizer"自动整理文件（遵循你的规则）
4. ✅ 使用"tapestry"提取内容并创建计划（保存到正确位置）
5. ✅ 所有现有skills已优化，融入新原则
6. ✅ 有完整的使用指南

---

**开始时间**: 2026-01-13
**预计完成**: 80分钟
**当前状态**: 准备阶段
