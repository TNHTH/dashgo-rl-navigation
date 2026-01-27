<#
.SYNOPSIS
    File Organizer PowerShell Examples
.DESCRIPTION
    Collection of PowerShell examples for file organization tasks
.NOTES
    Part of file-organizer skill
#>

# Example 1: Create Directory Structure
Write-Host "=== Example 1: Create Directory ===" -ForegroundColor Cyan

$docDir = "D:\cursor\file\Si Yuan\claude\"
New-Item -ItemType Directory -Force -Path $docDir | Out-Null
Write-Host "✓ Created: $docDir"

# Example 2: Move Files
Write-Host "`n=== Example 2: Move Files ===" -ForegroundColor Cyan

# Move all markdown files to docs directory
$files = Get-ChildItem -Filter "*.md" -File
foreach ($file in $files) {
    $dest = Join-Path $docDir $file.Name
    Move-Item $file.FullName $dest -Force
    Write-Host "✓ Moved: $($file.Name) → $docDir"
}

# Example 3: Clean Temporary Files
Write-Host "`n=== Example 3: Clean Temporary Files ===" -ForegroundColor Cyan

$tempDir = "D:\cursor\file\.claude-temp\"
if (Test-Path $tempDir) {
    $tempFiles = Get-ChildItem -Path $tempDir -File
    Write-Host "Found $($tempFiles.Count) temporary files"

    # Delete files older than 1 day
    $oldFiles = $tempFiles | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-1) }
    foreach ($file in $oldFiles) {
        Remove-Item $file.FullName -Force
        Write-Host "✓ Deleted: $($file.Name)"
    }
}

# Example 4: Rename Files with Pattern
Write-Host "`n=== Example 4: Batch Rename ===" -ForegroundColor Cyan

# Add date prefix to files
$pattern = "^(.*)\.md$"
$files = Get-ChildItem -Filter "*.md" -File
$date = Get-Date -Format "yyyy-MM-dd"

foreach ($file in $files) {
    if ($file.Name -match $pattern) {
        $newName = "{0}_{1}" -f $date, $file.Name
        Rename-Item $file.FullName -NewName $newName
        Write-Host "✓ Renamed: $($file.Name) → $newName"
    }
}

# Example 5: Find Duplicate Files
Write-Host "`n=== Example 5: Find Duplicates ===" -ForegroundColor Cyan

# Group files by size and name
$files = Get-ChildItem -Recurse -File
$groups = $files | Group-Object { $_.Length } | Where-Object { $_.Count -gt 1 }

foreach ($group in $groups) {
    if ($group.Count -gt 1) {
        Write-Host "`nDuplicate group (size: $($group.Name) bytes):"
        $group.Group | Select-Object -FirstProperty Name, FullName
    }
}

Write-Host "`n✅ All examples completed!"
