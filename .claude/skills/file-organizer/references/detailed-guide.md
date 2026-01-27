# File Organizer - 详细指南

## 核心概念

File Organizer skill帮助你智能组织文件，减少认知负担，保持数字工作空间整洁。

---

## 文件组织规则详解

### 文档分类系统

#### 非项目文档 → `D:\cursor\file\Si Yuan\claude\`

**包含内容**:
- 科普类文档（如投资策略、技术概念）
- 分析报告（行业分析、政策研究）
- 学习笔记类文档
- 任何非项目相关的独立文档

**判断流程**:
```
是代码文件？ → 项目目录
用户指定路径？ → 用用户指定路径
否则 → Si Yuan\claude\
```

#### 项目文档 → 项目目录

**包含内容**:
- 属于特定项目的代码
- 项目相关配置文件
- 项目文档（README、设计文档）
- 构建脚本和工具

**特点**:
- 随项目目录组织
- 不移动到全局文档目录

#### 临时文件 → `D:\cursor\file\.claude-temp\`

**包含内容**:
- tmpclaude-*-cwd临时文件
- 临时测试文件
- 临时脚本文件
- CSV分析结果（对话后可删除）

**清理策略**:
- 对话结束时自动清理
- 保留.claude-temp目录本身
- 从不提交到Git仓库

---

## 文件命名规范详解

### 中文文件名优势

- ✅ 本地化，易于识别
- ✅ 支持搜索（拼音、汉字）
- ✅ 符合用户习惯

### 命名格式

**完整格式**:
```
主题_类型_YYYY-MM-DD.md
```

**示例**:
- `低波投资策略_科普_2025-01-13.md`
- `阿根廷现状_分析_2026-01-12.md`
- `Claude-Skills使用指南_教程_2026-01-16.md`

**简化格式**:
```
主题_说明.md
```

**示例**:
- `低波投资策略科普.md`
- `阿根廷现状分析.md`

### 文件命名最佳实践

**DO**:
- ✅ 使用下划线分隔
- ✅ 包含日期便于排序
- ✅ 清晰描述内容
- ✅ 使用类型标识

**DON'T**:
- ❌ 使用空格（可能引起问题）
- ❌ 使用特殊字符（/ \ : * ? " < > |）
- ❌ 过长的文件名（>80字符）
- ❌ 含糊的名称

---

## 组织策略

### 按类型组织

```powershell
# 文档
*.md → Si Yuan\claude\
*.txt → Si Yuan\claude\

# 代码
*.py, *.js, *.ts → 项目目录

# 配置
*.json, *.yaml, *.toml → 项目目录

# 临时
*.tmp, *.temp → .claude-temp\
```

### 按日期组织

```powershell
# 归档旧文件
$oldDate = (Get-Date).AddMonths(-6)
Get-ChildItem -File | Where-Object { $_.LastWriteTime -lt $oldDate } | ForEach-Object {
    Move-Item $_.FullName "archive\$($_.Name)"
}
```

### 按项目组织

```
projects/
├── project-a/
│   ├── docs/
│   ├── src/
│   └── tests/
└── project-b/
    ├── docs/
    ├── src/
    └── tests/
```

---

## 常见场景处理

### 场景1: Downloads文件夹混乱

**问题**: Downloads文件夹堆积各种文件

**解决方案**:
1. 按文件类型分类（文档、安装包、压缩包）
2. 文档移动到Si Yuan\claude\
3. 安装包移动到软件目录
4. 压缩包解压后删除

### 场景2: 重复文件

**问题**: 相同文件散落不同位置

**解决方案**:
1. 使用文件hash识别重复
2. 保留最新版本
3. 删除旧重复文件
4. 建立符号链接（如需要）

### 场景3: 项目归档

**问题**: 旧项目需要归档

**解决方案**:
1. 创建archive/目录
2. 移动整个项目文件夹
3. 创建README说明归档原因
4. 更新项目索引

---

## PowerShell高级技巧

### 批量操作

```powershell
# 批量重命名
Get-ChildItem "*.txt" | ForEach-Object {
    $newName = $_.Name -replace 'old', 'new'
    Rename-Item $_.FullName $newName
}

# 批量移动
Get-ChildItem "*.md" | Move-Item -Destination "docs\"

# 批量删除
Get-ChildItem "*.log" -Recurse | Remove-Item -Force
```

### 文件筛选

```powershell
# 按大小筛选
Get-ChildItem | Where-Object { $_.Length -gt 1MB }

# 按日期筛选
Get-ChildItem | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) }

# 按类型筛选
Get-ChildItem -Include *.md,*.txt
```

### 安全操作

```powershell
# 验证后再删除
$files = Get-ChildItem "*.tmp"
Write-Host "Found $($files.Count) files to delete"
$confirm = Read-Host "Delete these files? (y/n)"
if ($confirm -eq 'y') {
    $files | Remove-Item -Force
}
```

---

## 错误处理

### 常见错误

**错误1: 路径不存在**
```powershell
if (-not (Test-Path $path)) {
    Write-Host "路径不存在: $path"
    # 创建路径
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}
```

**错误2: 文件已存在**
```powershell
if (Test-Path $dest) {
    Write-Host "文件已存在: $dest"
    $overwrite = Read-Host "覆盖? (y/n)"
    if ($overwrite -eq 'y') {
        Copy-Item $src $dest -Force
    }
}
```

**错误3: 权限不足**
```powershell
try {
    Remove-Item $file -Force
} catch {
    Write-Host "权限不足: $file"
    Write-Host "错误: $_"
}
```

---

## 维护计划

### 日常维护
- 每周清理Downloads文件夹
- 每月检查重复文件
- 每季度归档旧项目

### 自动化
- 对话结束时自动清理.claude-temp\
- 定期运行文件组织脚本
- 使用任务计划器自动化

---

**文档版本**: v1.0
**创建日期**: 2026-01-17
**用途**: File Organizer skill详细指南
