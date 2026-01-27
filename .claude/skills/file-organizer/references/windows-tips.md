# File Organizer - Windows特定提示

## Windows文件系统限制

### 路径长度限制
- **限制**: 260字符（MAX_PATH）
- **解决方案**: 使用长路径支持或缩短路径
- **最佳实践**: 保持在250字符以下

### 保留字符
- **保留**: / \ : * ? " < > |
- **替换**: 使用连字符-或下划线_
- **示例**: `file:name.txt` → `file_name.txt`

### 大小写不敏感
- Windows文件系统默认不区分大小写
- `FILE.TXT` 和 `file.txt` 视为相同
- 注意避免命名冲突

---

## PowerShell特定技巧

### 执行策略

**检查执行策略**:
```powershell
Get-ExecutionPolicy
```

**设置执行策略** (需要管理员权限):
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 中文编码

**PowerShell 7** (默认UTF-8):
```powershell
Get-Content $file  # UTF-8
```

**Windows PowerShell 5.1**:
```powershell
Get-Content $file -Encoding UTF8  # 需要指定
```

**写入文件**:
```powershell
# PS 7
"中文内容" | Out-File $file

# PS 5.1
"中文内容" | Out-File $file -Encoding UTF8
```

### 管理员权限

**检查是否管理员**:
```powershell
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if ($isAdmin) {
    Write-Host "运行管理员模式"
}
```

**以管理员运行**:
- 右键PowerShell图标
- 选择"以管理员身份运行"

---

## Windows资源管理器集成

### 右键菜单 (可选)

可以添加自定义右键菜单项：
- "组织到Si Yuan\claude\"
- "移动到.claude-temp\"
- "按类型分类"

### 库的使用

**创建自定义库**:
1. 打开文件资源管理器
2. 右键"库" → "新建"
3. 添加文件夹: `D:\cursor\file\Si Yuan\claude\`
4. 设置默认保存位置

---

## 性能优化

### 大量文件操作

**使用Measure-Command监控性能**:
```powershell
Measure-Command {
    Get-ChildItem -Recurse | Where-Object { $_.Length -gt 1MB }
}
```

**并行处理** (PowerShell 7):
```powershell
$files | ForEach-Object -Parallel {
    # 处理每个文件
} -ThrottleLimit 4
```

### 减少文件系统调用

**批量操作优于单文件操作**:
```powershell
# 差: 单个移动
foreach ($file in $files) {
    Move-Item $file $dest
}

# 好: 批量移动
Move-Item $files $dest
```

---

## Windows Defender排除

**添加文件夹到排除列表** (可选):
1. 打开Windows Security
2. 病毒和威胁防护 → 管理设置
3. 排除项 → 添加或删除排除项
4. 添加: `D:\cursor\file\.claude-temp\`

**好处**:
- 减少临时文件扫描
- 提升文件操作性能
- 降低磁盘活动

---

## 网络驱动器

### 映射网络驱动器

**映射驱动器**:
```powershell
Net-Use Z: \\server\share /persistent:yes
```

**访问网络文件**:
```powershell
# UNC路径
Copy-Item "\\server\share\file.txt" "D:\local\"

# 映射驱动器
Copy-Item "Z:\file.txt" "D:\local\"
```

---

## 备份策略

### Windows文件历史

**启用文件历史**:
1. 控制面板 → 文件历史
2. 选择驱动器
3. 排除.claude-temp\（可选）
4. 定期备份重要文件

### Robocopy备份

**示例备份脚本**:
```powershell
# 备份Si Yuan\claude\
$source = "D:\cursor\file\Si Yuan\claude\"
$dest = "E:\Backup\claude\"
robocopy $source $dest /E /Z /R:5 /W:5 /LOG:backup.log
```

**参数说明**:
- `/E`: 复制子目录（包括空目录）
- `/Z`: 使用重启模式
- `/R:5`: 重试5次
- `/W:5`: 等待5秒

---

## 故障排除

### 文件锁定

**问题**: 文件被占用无法移动

**解决方案**:
```powershell
# 查找占用文件的进程
$file = "D:\path\to\file.txt"
$handle = OpenFilesView /Query $file

# 或强制解锁（不推荐）
# 使用Process Explorer查找并终止进程
```

### 权限问题

**检查文件权限**:
```powershell
$acl = Get-Acl $file
$acl.Access | Format-Table IdentityReference, AccessControlType, FileSystemRights
```

**获取所有权** (需要管理员):
```powershell
takeown /f "$file"
icacls "$file" /grant "%USERNAME%:F"
```

---

**文档版本**: v1.0
**创建日期**: 2026-01-17
**用途**: Windows环境特定指南
