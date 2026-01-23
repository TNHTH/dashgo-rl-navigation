---
name: tapestry
description: Unified content extraction and action planning. Use when user says "tapestry <URL>", "weave <URL>", "help me plan <URL>", "extract and plan <URL>", "make this actionable <URL>", or similar phrases indicating they want to extract content and create an action plan. Automatically detects content type (YouTube video, article, PDF) and processes accordingly. Saves content to D:\cursor\file\Si Yuan\claude\extracted-content\ and plans to D:\cursor\file\Si Yuan\claude\plans\.
allowed-tools: Bash,Read,Write
---

# Tapestry: Unified Content Extraction + Action Planning

## 概述

从URL提取内容并自动创建Ship-Learn-Next行动计划。Windows优化环境。

**核心理念**: 不仅仅是消费内容，而是创建实施计划。将被动学习转变为主动构建。

---

## 何时使用

**激活关键词**:
- "tapestry [URL]"
- "weave [URL]"
- "help me plan [URL]"
- "extract and plan [URL]"
- "make this actionable [URL]"
- "turn [URL] into a plan"
- "learn and implement from this"

**关注词**: tapestry, weave, plan, actionable, extract and plan, make a plan, turn into action

---

## 文件组织

```
Content files → D:\cursor\file\Si Yuan\claude\extracted-content\
Plan files    → D:\cursor\file\Si Yuan\claude\plans\
Temp files    → D:\cursor\file\.claude-temp\
```

---

## 工作流程

### 5步完整流程

1. **检测URL类型** (YouTube, Article, PDF)
2. **提取内容** (使用相应Windows工具)
3. **保存到正确位置** (遵循文件组织规则)
4. **创建行动计划** (Ship-Learn-Next方法论)
5. **展示摘要** (给用户)

---

## URL检测逻辑

### YouTube Videos

**检测模式**:
- `youtube.com/watch?v=`
- `youtu.be/`
- `youtube.com/shorts/`
- `m.youtube.com/watch?v=`

**提取工具**: yt-dlp (Windows)

### Web Articles/Blog Posts

**检测模式**:
- `http://` or `https://`
- NOT YouTube, NOT PDF
- Medium, Substack, dev.to等
- 任何HTML页面

**提取工具**: reader or trafilatura (Windows)

### PDF Documents

**检测模式**:
- URL ends with `.pdf`
- `Content-Type: application/pdf`

**提取工具**: pdftotext (Windows)

---

## Ship-Learn-Next方法论

### 核心概念

**Extract → Plan → Ship → Learn → Next**

不仅仅是消费内容，而是：
1. **提取**: 获取核心知识点
2. **计划**: 定义4-8周探索任务
3. **实施**: Rep 1本周可交付
4. **学习**: 迭代中学习
5. **下一步**: Reps 2-5渐进改进

### 行动计划结构

```markdown
## 🎯 Your Quest
[一行总结你要构建什么]

## 📚 Key Learnings from Content
- [可操作课程1]
- [可操作课程2]
- [可操作课程3]

## 📍 Rep 1: Ship This Week
**Goal**: [本周交付什么]
**Timeline**: 本周
**Definition of Done**: [如何知道完成]

## 🔮 Reps 2-5: Progressive Iterations
### Rep 2
**Goal**: [下次迭代]
**Timeline**: 第2周

[继续Reps 3-5]

---
**Next Action**: 你何时交付Rep 1?
```

---

## 最佳实践

### 执行时
- ✅ 显示检测结果 ("📍 Detected: youtube")
- ✅ 展示每步进度
- ✅ 保存content AND plan到正确位置
- ✅ 显示提取内容预览 (前10行)
- ✅ 自动创建计划 (不询问)
- ✅ 结束时展示清晰摘要
- ✅ 询问承诺问题："你何时交付Rep 1?"

### Windows环境
- ✅ 使用PowerShell 7+
- ✅ 始终使用UTF-8编码
- ✅ 引用带空格路径: `"D:\path\with spaces\"`
- ✅ 使用 `Get-Command` 检查工具是否存在
- ✅ 使用 `try/catch` 处理网络操作

### 文件命名
- ✅ 替换无效字符: `/ \ : * ? " < > |` → `-`
- ✅ 限制文件名80字符
- ✅ 限制路径250字符 (Windows限制)
- ✅ 使用日期标识: `YYYY-MM-DD`

---

## 错误处理

### 常见问题

**1. 不支持的URL类型**
- 尝试文章提取作为fallback
- 失败则: "无法从此URL类型提取内容"

**2. 未提取到内容**
- 检查URL是否可访问
- 尝试备用提取方法
- 通知用户: "提取失败。URL可能需要认证。"

**3. 工具未安装**
- 提供安装命令
- YouTube: `winget install yt-dlp`
- Articles: `pip install trafilatura`
- PDFs: `choco install poppler`
- 使用fallback方法（如可用）

**4. 无效文件名字符**
- 自动替换无效字符
- 限制文件名长度
- 去除空白

---

## 依赖工具

**YouTube提取**:
- yt-dlp: `winget install yt-dlp`
- Python 3: `winget install Python`

**文章提取**:
- reader: `npm install -g @mozilla/readability-cli`
- OR trafilatura: `pip install trafilatura`
- Fallback: Invoke-WebRequest (内建)

**PDF提取**:
- poppler: `choco install poppler`
- OR仅下载PDF不提取文本

**计划创建**:
- 无额外需求 (使用内建工具)

---

## Optional Reading (按需加载)

深入了解实现细节：
- `.claude/skills/tapestry/scripts/extract-youtube.ps1` - YouTube提取脚本
- `.claude/skills/tapestry/scripts/extract-article.ps1` - 文章提取脚本
- `.claude/skills/tapestry/scripts/extract-pdf.ps1` - PDF提取脚本
- `.claude/skills/tapestry/scripts/tapestry.ps1` - 完整工作流脚本
- `.claude/skills/tapestry/references/windows-guide.md` - Windows详细指南

历史版本归档在 `archive/tapestry-v1.0.md`

---

**Version**: v2.0 (Lean Runtime)
**Last Updated**: 2026-01-17
**Philosophy**: Extract → Plan → Ship → Learn → Next
