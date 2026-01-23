# Claude Code CLI 完整迁移方案

> **创建时间**: 2026-01-22 16:30:00
> **适用场景**: 转移到另一个电脑/系统/设备
> **预计时间**: 15-30分钟

---

## 一、迁移前准备

### 1. 确认当前系统组件

**需要迁移的内容**：
```
D:\cursor\file\
├── .claude/                    # Claude配置（核心）
│   ├── skills/                 # 全局skills
│   ├── rules/                  # 动态规则
│   ├── hooks/                  # 自动化钩子
│   └── settings.local.json     # 本地设置
├── multi-agent-system/         # 多智能体系统
│   └── .claude/skills/         # 项目skills
└── Si Yuan/claude/             # 文档和知识库
```

**不需要迁移的内容**：
- `node_modules/`
- `.git/`（除非要迁移整个仓库）
- 临时文件 `.claude-temp/`

### 2. 确认目标系统

| 场景 | 说明 | 难度 |
|------|------|------|
| **场景A**: 另一台Windows电脑 | 最常见 | ⭐ |
| **场景B**: macOS电脑 | 需要路径调整 | ⭐⭐ |
| **场景C**: Linux电脑 | 需要路径调整 | ⭐⭐ |
| **场景D**: 另一个AI工具（Cursor） | 只迁移配置和文档 | ⭐⭐⭐ |

---

## 二、场景A：迁移到另一台Windows电脑（推荐流程）

### Step 1: 导出配置包（在当前电脑）

#### 创建导出脚本

```powershell
# export-claude-config.ps1
# 运行此脚本导出Claude Code配置

$configPath = "D:\cursor\file"
$exportPath = "$HOME\Desktop\Claude-Code-Config-Backup"
$date = Get-Date -Format "yyyy-MM-dd-HH-mm"

# 创建导出目录
New-Item -ItemType Directory -Path "$exportPath\$date" -Force | Out-Null

# 1. 导出.claude配置
Write-Host "正在导出.claude配置..."
Copy-Item -Path "$configPath\.claude" `
          -Destination "$exportPath\$date\.claude" `
          -Recurse -Force

# 2. 导出multi-agent-system
Write-Host "正在导出multi-agent-system..."
Copy-Item -Path "$configPath\multi-agent-system" `
          -Destination "$exportPath\$date\multi-agent-system" `
          -Recurse -Force

# 3. 导出Si Yuan/claude文档
Write-Host "正在导出Si Yuan/claude文档..."
Copy-Item -Path "$configPath\Si Yuan\claude" `
          -Destination "$exportPath\$date\docs" `
          -Recurse -Force

# 4. 创建迁移清单
$manifest = @"
# Claude Code 配置迁移清单

> 导出时间: $date
> 源路径: $configPath
> 目标系统: [待填写]

## 已导出内容

### 1. .claude/ 配置
- skills/ (全局skills)
- rules/ (动态规则)
- hooks/ (自动化钩子)
- settings.local.json (本地设置)

### 2. multi-agent-system/
- agents/ (8个agents)
- .claude/skills/ (项目skills)
- shared/ (共享工具指南)

### 3. docs/ (Si Yuan/claude)
- 个人分析/
- 使用指南/
- 分析报告/
- 会议记录/
- 科普学习/
- 系统更新/
- 系统优化/

## 迁移步骤

1. 在新电脑上安装Claude Code CLI
2. 将此文件夹复制到新电脑
3. 运行import-claude-config.ps1导入配置
4. 验证skills和agents是否正常工作

## 注意事项

- Windows路径格式：使用反斜杠 `\`
- PowerShell命令：确认执行策略 `Set-ExecutionPolicy RemoteSigned`
- Git仓库：如需保留版本历史，单独迁移.git文件夹
"@

Set-Content -Path "$exportPath\$date\README.md" -Value $manifest

Write-Host ""
Write-Host "✅ 配置导出完成！" -ForegroundColor Green
Write-Host "📁 导出位置: $exportPath\$date"
Write-Host "📋 请查看README.md了解迁移步骤"
```

**运行导出**：
```powershell
# 在PowerShell中运行
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\export-claude-config.ps1
```

---

### Step 2: 传输到目标电脑

**选项1：云盘同步**（推荐）
```
上传到百度网盘/OneDrive/Google Drive
→ 在目标电脑下载
```

**选项2：U盘拷贝**
```
复制到U盘
→ 插入目标电脑
→ 复制到桌面
```

**选项3：局域网共享**
```
# 在当前电脑设置共享
net share ClaudeConfig=$exportPath /grant:Everyone,FULL

# 在目标电脑访问
\\[当前电脑IP]\ClaudeConfig
```

---

### Step 3: 导入配置（在目标电脑）

#### 安装Claude Code CLI

```powershell
# 1. 安装Node.js（如果未安装）
# 下载：https://nodejs.org/

# 2. 安装Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 3. 验证安装
claude --version
```

#### 创建导入脚本

```powershell
# import-claude-config.ps1
# 在目标电脑运行此脚本导入配置

param(
    [Parameter(Mandatory=$true)]
    [string]$SourcePath  # 导出配置的路径，例如: C:\Users\YourName\Desktop\Claude-Code-Config-Backup\2026-01-22-16-30
)

# 目标路径
$targetPath = "D:\cursor\file"

# 检查源路径是否存在
if (-not (Test-Path $SourcePath)) {
    Write-Host "❌ 错误：源路径不存在 - $SourcePath" -ForegroundColor Red
    exit 1
}

# 创建目标目录
New-Item -ItemType Directory -Path $targetPath -Force | Out-Null

# 1. 导入.claude配置
Write-Host "正在导入.claude配置..."
Copy-Item -Path "$SourcePath\.claude" `
          -Destination "$targetPath\.claude" `
          -Recurse -Force

# 2. 导入multi-agent-system
Write-Host "正在导入multi-agent-system..."
Copy-Item -Path "$SourcePath\multi-agent-system" `
          -Destination "$targetPath\multi-agent-system" `
          -Recurse -Force

# 3. 导入文档
Write-Host "正在导入文档..."
Copy-Item -Path "$SourcePath\docs" `
          -Destination "$targetPath\Si Yuan\claude" `
          -Recurse -Force

# 4. 验证导入
Write-Host ""
Write-Host "✅ 配置导入完成！" -ForegroundColor Green
Write-Host "📁 目标位置: $targetPath"
Write-Host ""
Write-Host "请验证以下内容：" -ForegroundColor Yellow
Write-Host "1. Skills: Test-Path '$targetPath\.claude\skills'"
Write-Host "2. Agents: Test-Path '$targetPath\multi-agent-system\agents'"
Write-Host "3. Rules: Test-Path '$targetPath\.claude\rules'"
Write-Host "4. Docs: Test-Path '$targetPath\Si Yuan\claude'"
```

**运行导入**：
```powershell
# 在目标电脑PowerShell中运行
.\import-claude-config.ps1 -SourcePath "C:\Users\YourName\Desktop\Claude-Code-Config-Backup\2026-01-22-16-30"
```

---

### Step 4: 验证配置

```powershell
# 在目标电脑验证

# 1. 检查文件是否存在
Test-Path "D:\cursor\file\.claude\skills"
Test-Path "D:\cursor\file\multi-agent-system\agents"
Test-Path "D:\cursor\file\.claude\rules\dynamic_rules.md"

# 2. 启动Claude Code
cd D:\cursor\file
claude

# 3. 测试skills
# 在Claude Code中输入：
# /brainstorming
# /kaizen

# 4. 测试agents
# 在Claude Code中输入：
# "请使用architect agent帮我设计系统架构"
```

---

## 三、场景B/C：迁移到macOS/Linux

### 主要差异

| 项目 | Windows | macOS/Linux |
|------|---------|-------------|
| 路径分隔符 | `\` | `/` |
| 配置路径 | `D:\cursor\file` | `~/cursor-file` 或 `~/projects/cursor-file` |
| Shell命令 | PowerShell | Bash/Zsh |
| Git路径处理 | 自动转换 | 需要手动设置 |

### 迁移脚本（macOS/Linux）

```bash
#!/bin/bash
# import-claude-config.sh
# 在macOS/Linux上运行

SOURCE_PATH="$1"  # 导出配置的路径
TARGET_PATH="$HOME/cursor-file"

# 检查源路径
if [ ! -d "$SOURCE_PATH" ]; then
    echo "❌ 错误：源路径不存在 - $SOURCE_PATH"
    exit 1
fi

# 创建目标目录
mkdir -p "$TARGET_PATH"

# 导入配置
echo "正在导入.claude配置..."
cp -r "$SOURCE_PATH/.claude" "$TARGET_PATH/"

echo "正在导入multi-agent-system..."
cp -r "$SOURCE_PATH/multi-agent-system" "$TARGET_PATH/"

echo "正在导入文档..."
mkdir -p "$TARGET_PATH/Si Yuan"
cp -r "$SOURCE_PATH/docs" "$TARGET_PATH/Si Yuan/claude"

# 验证
echo ""
echo "✅ 配置导入完成！"
echo "📁 目标位置: $TARGET_PATH"
echo ""
echo "请验证："
echo "1. ls -la '$TARGET_PATH/.claude/skills'"
echo "2. ls -la '$TARGET_PATH/multi-agent-system/agents'"
echo "3. ls -la '$TARGET_PATH/.claude/rules'"
```

**路径转换**（如果需要）：

```bash
# 将Windows路径转换为Unix路径
find . -name "*.md" -type f -exec sed -i 's/D:\\cursor\\file\\/~/cursor-file\//g' {} +
```

---

## 四、场景D：迁移到其他AI工具（Cursor/Windsurf）

### 可迁移的内容

✅ **可以迁移**：
- `Si Yuan/claude/` - 所有文档和知识库
- `.claude/rules/` - 编码规范、动态规则
- `multi-agent-system/agents/` - Agent提示词（参考）
- `.claude/skills/` - Skills概念（需要适配）

❌ **不能迁移**：
- `.claude/skills/*.json` - Skills格式不兼容
- `.claude/hooks/` - Hooks机制不同
- `.claude/settings.local.json` - 设置格式不同

### 迁移到Cursor的步骤

```
1. 安装Cursor
2. 复制以下内容到Cursor项目：
   ├── Si Yuan/claude/          # 文档
   ├── .cursor/rules/            # 编码规范（转换后）
   └── prompts/                  # Agent提示词（手动适配）

3. 在Cursor中创建Rules：
   - 将coding-style.md内容复制到Cursor的.claude/rules/
   - 将dynamic_rules.md内容手动转换为Cursor格式

4. 在Cursor中创建Prompts：
   - 复制architect.prompt.md到Cursor的Prompt库
   - 复制code-reviewer.prompt.md到Cursor的Prompt库
```

---

## 五、高级场景：保留Git历史

### 方法1：推送完整仓库到GitHub（已完成）

```bash
# 在当前电脑
cd "D:\cursor\file"
git remote add origin https://github.com/TNHTH/file-workspace.git
git push -u origin main

# 在目标电脑
git clone https://github.com/TNHTH/file-workspace.git "D:\cursor\file"
```

### 方法2：打包Git仓库

```bash
# 在当前电脑
cd "D:\cursor\file"
git bundle create claude-code.bundle --all

# 传输claude-code.bundle到目标电脑

# 在目标电脑
git clone claude-code.bundle "D:\cursor\file"
```

---

## 六、验证清单

### 基础验证

- [ ] Claude Code CLI可以正常启动
- [ ] 所有skills可以正常调用（/brainstorming, /kaizen等）
- [ ] agents可以正常工作
- [ ] 文档路径正确
- [ ] 动态规则已加载

### 高级验证

- [ ] Git提交历史完整（如果迁移了.git）
- [ ] Hooks正常运行（清理临时文件）
- [ ] 所有文档可以正常打开
- [ ] 代码审查功能正常

---

## 七、常见问题

### Q1: 导入后skills不工作？

**检查**：
```powershell
# 确认skills文件夹存在
Test-Path "D:\cursor\file\.claude\skills"

# 确认SKILL.md文件存在
Get-ChildItem "D:\cursor\file\.claude\skills" -Recurse -Filter "SKILL.md"
```

**修复**：
```powershell
# 重新创建skills结构
Get-ChildItem "源路径\.claude\skills" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination "D:\cursor\file\.claude\skills\" -Recurse
}
```

### Q2: 路径错误怎么办？

**Windows路径**：
```
D:\cursor\file\Si Yuan\claude
```

**macOS/Linux路径**：
```
~/cursor-file/Si Yuan/claude
```

**替换路径**：
```powershell
# 批量替换文档中的路径
Get-ChildItem "D:\cursor\file\Si Yuan\claude" -Recurse -Filter "*.md" | ForEach-Object {
    (Get-Content $_.FullName) -replace 'D:\\cursor\\file\\', '~/cursor-file/' | Set-Content $_.FullName
}
```

### Q3: Git历史丢失？

**原因**：只复制了文件，没有复制.git文件夹

**解决**：
```bash
# 方法1：从GitHub克隆（如果已推送）
git clone https://github.com/TNHTH/file-workspace.git "D:\cursor\file"

# 方法2：使用git bundle（如果有备份）
git clone claude-code.bundle "D:\cursor\file"
```

---

## 八、推荐方案

### 最简单：云盘 + 手动导入

```
1. 在当前电脑运行 export-claude-config.ps1
2. 上传导出文件夹到百度网盘/OneDrive
3. 在目标电脑下载
4. 运行 import-claude-config.ps1
5. 验证配置
```

**时间**: 15分钟
**难度**: ⭐
**可靠性**: ⭐⭐⭐⭐⭐

---

### 最完整：GitHub + 完整克隆

```
1. 推送当前仓库到GitHub（已完成）
2. 在目标电脑: git clone https://github.com/TNHTH/file-workspace.git
3. 验证配置
```

**时间**: 10分钟
**难度**: ⭐
**可靠性**: ⭐⭐⭐⭐⭐
**保留历史**: ✅

---

## 九、后续维护

### 定期同步配置

```powershell
# 每月同步一次到GitHub
cd "D:\cursor\file"
git add .
git commit -m "Monthly sync: $(Get-Date -Format 'yyyy-MM-dd')"
git push origin main
```

### 多设备同步

```bash
# 在设备A提交并推送
git push origin main

# 在设备B拉取更新
git pull origin main
```

---

**你想要哪种迁移方案？**
- **场景A**: 另一台Windows电脑（最常见）
- **场景B/C**: macOS/Linux
- **场景D**: 其他AI工具（Cursor）
- **完整方案**: GitHub同步（推荐）

或者告诉我具体的目标系统，我可以提供定制化方案。
