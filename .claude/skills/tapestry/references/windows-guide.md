# Tapestry Windows详细指南

## Windows环境要求

### PowerShell版本
- **推荐**: PowerShell 7+ (`pwsh`)
- **兼容**: Windows PowerShell 5.1+ (`powershell`)
- **检查**: `$PSVersionTable.PSVersion`

### 字符编码
- **标准**: UTF-8
- **PowerShell 7**: 默认UTF-8
- **Windows PowerShell**: 使用 `-Encoding UTF8` 参数

## 工具安装

### YouTube转录提取
```powershell
# 方法1: winget (推荐)
winget install yt-dlp

# 方法2: pip
pip install yt-dlp

# 验证安装
yt-dlp --version
```

### 文章内容提取
```powershell
# 方法1: reader (npm) - 推荐
npm install -g @mozilla/readability-cli

# 方法2: trafilatura (pip)
pip install trafilatura

# 验证
reader --help
# 或
python -c "import trafilatura; print(trafilatura.__version__)"
```

### PDF文本提取
```powershell
# 使用Chocolate安装poppler
choco install poppler

# 验证
pdftotext -v
```

## Windows特定注意事项

### 路径处理
```powershell
# 正确: 引用带空格的路径
$PATH = "D:\cursor\file\Si Yuan\claude\"

# 错误: 不引用带空格的路径
$PATH = D:\cursor\file\Si Yuan\claude\  # 会失败
```

### 文件命名限制
**Windows保留字符**: `/ \ : * ? " < > |`

**自动替换策略**:
```powershell
$CLEAN_NAME = $FILENAME -replace '[/\\:*?"<>|]', '-'
$CLEAN_NAME = $CLEAN_NAME.Substring(0, [Math]::Min(80, $CLEAN_NAME.Length))
```

### 路径长度限制
- **限制**: 260字符
- **策略**: 保持在250字符以下
- **长路径支持**: Windows 10+可启用长路径（组策略）

## 错误处理

### 工具检查
```powershell
$toolExists = Get-Command <tool-name> -ErrorAction SilentlyContinue
if (-not $toolExists) {
    Write-Host "Tool not found. Install with: <install-command>"
    exit 1
}
```

### 网络错误
```powershell
try {
    $response = Invoke-WebRequest -Uri $URL
} catch {
    Write-Host "❌ Network error: $_"
    exit 1
}
```

### 编码错误
```powershell
# PowerShell 7 (默认UTF-8)
Get-Content $file

# Windows PowerShell (需要指定)
Get-Content $file -Encoding UTF8
```

## 性能优化

### 并行处理
```powershell
# PowerShell 7+ 支持并行
$urls = @("url1", "url2", "url3")
$urls | ForEach-Object -Parallel {
    # 处理每个URL
} -ThrottleLimit 3
```

### 缓存策略
```powershell
# 检查文件是否已存在
if (Test-Path $CACHE_FILE) {
    $CACHE_AGE = (Get-Date) - (Get-Item $CACHE_FILE).LastWriteTime
    if ($CACHE_AGE.TotalDays -lt 7) {
        Write-Host "Using cached version"
        return
    }
}
```

## 调试技巧

### 详细输出
```powershell
# 启用详细输出
$VerbosePreference = "Continue"
Write-Verbose "Detailed debug info"

# 或直接输出变量
Write-Host "Variable value: $VAR"
```

### 错误跟踪
```powershell
# 显示完整错误信息
$ErrorActionPreference = "Stop"
try {
    # 代码
} catch {
    Write-Host "Error: $_"
    Write-Host "Stack: $($_.ScriptStackTrace)"
}
```

### 逐步执行
```powershell
# 暂停执行，等待用户输入
Read-Host "Press Enter to continue..."

# 或设置断点（ISE/VSCode）
Set-PSBreakpoint -Script $script -Line 42
```

## 文件组织

### 目录结构
```
D:\cursor\file\
├── Si Yuan\
│   └── claude\
│       ├── extracted-content\  # 提取的内容
│       └── plans\              # 行动计划
└── .claude-temp\               # 临时文件
```

### 自动创建目录
```powershell
$DIR = "D:\cursor\file\Si Yuan\claude\extracted-content\"
New-Item -ItemType Directory -Force -Path $DIR | Out-Null
```

## 安全注意事项

### URL验证
```powershell
# 基本URL格式验证
if ($URL -notmatch '^https?://') {
    Write-Host "❌ Invalid URL format"
    exit 1
}
```

### 文件覆盖保护
```powershell
# 检查文件是否存在
if (Test-Path $OUTPUT_FILE) {
    $OVERWRITE = Read-Host "File exists. Overwrite? (y/n)"
    if ($OVERWRITE -ne 'y') {
        exit 0
    }
}
```

### 临时文件清理
```powershell
# 清理临时文件
Get-ChildItem -Filter "temp_*" | Remove-Item -Force

# 或使用try/finally确保清理
try {
    # 创建临时文件
    $TEMP_FILE = "temp_$(Get-Date -Format 'yyyyMMddHHmmss')"
} finally {
    # 清理
    if (Test-Path $TEMP_FILE) {
        Remove-Item $TEMP_FILE -Force
    }
}
```

---

**文档版本**: v1.0
**创建日期**: 2026-01-17
**来源**: Tapestry v1.0 Windows特定内容提取
