---
name: file-organizer
description: Intelligently organizes your files and folders across your computer by understanding context, finding duplicates, suggesting better structures, and automating cleanup tasks. Follows your file organization rules (D:\cursor\file\Si Yuan\claude\ for docs, .claude-temp\ for temp files). Reduces cognitive load and keeps your digital workspace tidy without manual effort.
---

# File Organizer

智能文件组织助手，帮助维护清晰、逻辑化的文件结构。

**集成文件组织规则**:
- 非项目文档 → `D:\cursor\file\Si Yuan\claude\`
- 临时文件 → `D:\cursor\file\.claude-temp\`
- 项目文件 → 各自项目目录

---

## 何时使用

- Downloads文件夹混乱
- 文件散落各处找不到
- 有重复文件占用空间
- 文件夹结构不合理
- 需要建立更好的组织习惯
- 新项目需要良好结构
- 清理和归档旧项目

---

## 核心功能

1. **分析当前结构**: 审查文件夹和文件
2. **查找重复**: 识别系统中的重复文件
3. **建议组织方案**: 基于内容提出逻辑结构
4. **自动化清理**: 经批准后移动、重命名、组织文件
5. **维护上下文**: 基于文件类型、日期、内容智能决策
6. **减少杂乱**: 识别可能不需要的旧文件

---

## 文件组织规则

### 文档存放规则

```
✅ 非项目文档 → D:\cursor\file\Si Yuan\claude\
   - 科普类文档
   - 分析报告
   - 学习笔记类文档
   - 任何非项目相关的独立文档

✅ 项目文档 → 项目目录
   - 用户明确指定路径
   - 属于特定项目

✅ 临时文件 → D:\cursor\file\.claude-temp\
   - tmpclaude-*-cwd文件
   - 临时测试文件
   - 临时脚本文件
```

### 文件命名规范

- ✅ 使用中文文件名便于识别
- ✅ 格式: `主题_类型_YYYY-MM-DD.md`
  - 示例: `低波投资策略_科普_2025-01-13.md`
  - 或简化: `主题_说明.md`

### 清理策略

- 对话结束时: 可清理`.claude-temp\`
- 保留`.claude-temp\`文件夹本身
- 从不提交到Git仓库

---

## 快速检查清单

### 生成新文档时
```
□ 是项目文件？
  - 是 → 存放到项目目录
  - 否 → 继续检查

□ 用户明确指定路径？
  - 是 → 使用用户指定路径
  - 否 → 存放到 Si Yuan\claude\

□ 检查目标目录是否存在
  - 不存在 → 创建目录

□ 使用规范的文件命名
```

### 清理临时文件时
```
□ 确认是否为临时文件
□ 存放到 .claude-temp/ 目录
□ 对话结束后提醒用户清理
```

### 文件组织时
```
□ 分析当前文件结构
□ 识别重复文件
□ 提出组织方案
□ 获得用户批准
□ 执行文件移动/重命名
```

---

## 使用方式

### 基本工作流

```powershell
# 1. 进入项目目录
cd D:\cursor\file

# 2. 请求组织文件
"请帮我组织这些文件"

# 3. AI分析并建议方案
# 4. 用户批准
# 5. 自动执行组织
```

### 常用命令

**分析文件夹**:
```powershell
"分析Downloads文件夹并建议组织方案"
```

**查找重复文件**:
```powershell
"查找重复的PDF文件"
```

**清理临时文件**:
```powershell
"清理.claude-temp目录"
```

**组织项目文件**:
```powershell
"按类型组织当前目录的文件"
```

---

## 最佳实践

### 文档生成
- ✅ 自动应用文件组织规则
- ✅ 使用中文文件名
- ✅ 添加日期标识
- ✅ 分类存放

### 临时文件
- ✅ 统一存放.claude-temp\
- ✅ 对话结束后清理
- ✅ 不提交到Git

### 文件命名
- ✅ 清晰描述内容
- ✅ 使用分隔符"_"
- ✅ 包含类型标识
- ✅ 避免特殊字符

---

## PowerShell示例

详见: `scripts/examples.ps1`

基本操作:
```powershell
# 创建目录
New-Item -ItemType Directory -Force -Path "Si Yuan\claude\"

# 移动文件
Move-Item "file.txt" "Si Yuan\claude\file.txt"

# 清理临时文件
Get-ChildItem ".claude-temp\*.txt" | Remove-Item
```

---

## Optional Reading (按需加载)

深入了解:
- `.claude/skills/file-organizer/scripts/examples.ps1` - PowerShell示例
- `.claude/skills/file-organizer/references/detailed-guide.md` - 详细指南
- `.claude/skills/file-organizer/references/windows-tips.md` - Windows特定提示

历史版本归档在 `archive/file-organizer-v1.0.md`

---

**Version**: v2.0 (Lean Runtime)
**Last Updated**: 2026-01-17
**Integrated Rules**: file-organization.md
