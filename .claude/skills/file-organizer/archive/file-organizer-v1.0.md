---
name: file-organizer
description: Intelligently organizes your files and folders across your computer by understanding context, finding duplicates, suggesting better structures, and automating cleanup tasks. Follows your file organization rules (D:\cursor\file\Si Yuan\claude\ for docs, .claude-temp\ for temp files). Reduces cognitive load and keeps your digital workspace tidy without manual effort.
---

# File Organizer

This skill acts as your personal organization assistant, helping you maintain a clean, logical file structure across your computer without the mental overhead of constant manual organization.

**Integrated with your file organization rules:**
- Non-project documents → `D:\cursor\file\Si Yuan\claude\`
- Temporary files → `D:\cursor\file\.claude-temp\`
- Project files → Respective project directories

## When to Use This Skill

- Your Downloads folder is a chaotic mess
- You can't find files because they're scattered everywhere
- You have duplicate files taking up space
- Your folder structure doesn't make sense anymore
- You want to establish better organization habits
- You're starting a new project and need a good structure
- You're cleaning up before archiving old projects

## What This Skill Does

1. **Analyzes Current Structure**: Reviews your folders and files to understand what you have
2. **Finds Duplicates**: Identifies duplicate files across your system
3. **Suggests Organization**: Proposes logical folder structures based on your content
4. **Automates Cleanup**: Moves, renames, and organizes files with your approval
5. **Maintains Context**: Makes smart decisions based on file types, dates, and content
6. **Reduces Clutter**: Identifies old files you probably don't need anymore

## Your File Organization Rules

This skill follows your established rules from `file-organization.md`:

### Document Storage Rules

```
✅ Non-Project Documents → D:\cursor\file\Si Yuan\claude\
   - 科普类文档
   - 分析报告
   - 学习笔记类文档
   - 任何非项目相关的独立文档

✅ Project Documents → Project Directory
   - User explicitly specified path
   - Belongs to specific project

✅ Temporary Files → D:\cursor\file\.claude-temp\
   - tmpclaude-*-cwd files
   - Temporary test files
   - Temporary script files
```

### File Naming Conventions

- Use Chinese filenames for better recognition
- Format: `主题_类型_YYYY-MM-DD.md`
  - Example: `低波投资策略_科普_2025-01-13.md`
  - Example: `阿根廷现状_分析_2026-01-12.md`
- Or simpler: `主题_说明.md`

### Cleanup Strategy

- Conversation ends: Can clean up `.claude-temp\`
- Keep `.claude-temp\` folder itself
- Never commit to Git repository

## How to Use

### From Your Home Directory

```powershell
cd D:\cursor\file
```

Then ask Claude:
```
Help me organize my Downloads folder
```

```
Find duplicate files in my Documents folder
```

```
Review my project directories and suggest improvements
```

### Specific Organization Tasks

```
Organize these downloads into proper folders based on what they are
```

```
Find duplicate files and help me decide which to keep
```

```
Clean up old files I haven't touched in 6+ months
```

```
Create a better folder structure for my [work/projects/photos/etc]
```

## Instructions

When a user requests file organization help:

### 1. Understand the Scope

Ask clarifying questions:
- Which directory needs organization? (Downloads, Documents, entire folder?)
- What's the main problem? (Can't find things, duplicates, too messy, no structure?)
- Any files or folders to avoid? (Current projects, sensitive data?)
- How aggressively to organize? (Conservative vs. comprehensive cleanup)

### 2. Analyze Current State

Review the target directory:
```powershell
# Get overview of current structure
Get-ChildItem -Force "target_directory"

# Check file types
Get-ChildItem -Recurse -File "target_directory" | Group-Object Extension | Sort-Object Count -Descending

# Identify largest files
Get-ChildItem -Recurse -File "target_directory" | Sort-Object Length -Descending | Select-Object -First 20

# Count by file type
Get-ChildItem -Recurse -File "target_directory" | Group-Object Extension
```

Summarize findings:
- Total files and folders
- File type breakdown
- Size distribution
- Date ranges
- Obvious organization issues

### 3. Identify Organization Patterns

Based on the files, determine logical groupings:

**By Type**:
- Documents (PDFs, DOCX, TXT, MD)
- Images (JPG, PNG, SVG)
- Videos (MP4, MOV)
- Archives (ZIP, TAR, RAR)
- Code/Projects (directories with code)
- Spreadsheets (XLSX, CSV)
- Presentations (PPTX, KEY)

**By Purpose**:
- Work vs. Personal
- Active vs. Archive
- Project-specific
- Reference materials
- Temporary/scratch files

**By Your Rules**:
- Non-project docs → `Si Yuan\claude\`
- Project docs → Project directories
- Temporary → `.claude-temp\`

### 4. Find Duplicates

When requested, search for duplicates:
```powershell
# Find files with same name
Get-ChildItem -Recurse -File | Group-Object Name | Where-Object { $_.Count -gt 1 }

# Find files with same size (potential duplicates)
Get-ChildItem -Recurse -File | Group-Object Length | Where-Object { $_.Count -gt 1 }
```

For each set of duplicates:
- Show all file paths
- Display sizes and modification dates
- Recommend which to keep (usually newest or best-named)
- **Important**: Always ask for confirmation before deleting

### 5. Propose Organization Plan

Present a clear plan before making changes:

```markdown
# Organization Plan for [Directory]

## Current State
- X files across Y folders
- [Size] total
- File types: [breakdown]
- Issues: [list problems]

## Proposed Structure

Following your file organization rules:

```
Directory/
├── Projects/           → Keep in project directories
├── Si Yuan/
│   └── claude/        → Non-project documents
│       ├──科普/
│       ├──分析/
│       └──笔记/
└── .claude-temp/      → Temporary files
```

## Changes I'll Make

1. **Create new folders**: [list]
2. **Move files**:
   - X non-project docs → `Si Yuan\claude\`
   - Y project files → respective project directories
   - Z temp files → `.claude-temp\`
3. **Rename files**: [any renaming patterns following `主题_类型_日期.md`]
4. **Delete**: [duplicates or trash files]

## Files Needing Your Decision

- [List any files you're unsure about]

Ready to proceed? (yes/no/modify)
```

### 6. Execute Organization

After approval, organize systematically:

```powershell
# Create folder structure
New-Item -ItemType Directory -Force -Path "path\to\new\folders"

# Move files with clear logging
Move-Item "old\path\file.pdf" "new\path\file.pdf"

# Rename files with consistent patterns
# Example: "2026-01-13 - 低波投资策略科普.md"
```

**Important Rules**:
- Always confirm before deleting anything
- Log all moves for potential undo
- Preserve original modification dates
- Handle filename conflicts gracefully
- Stop and ask if you encounter unexpected situations
- **Follow file organization rules**: Non-project docs → `Si Yuan\claude\`

### 7. Provide Summary and Maintenance Tips

After organizing:

```markdown
# Organization Complete! ✨

## What Changed

- Created [X] new folders
- Organized [Y] files
- Freed [Z] GB by removing duplicates
- Archived [W] old files

## New Structure

[Show the new folder tree]

## Maintenance Tips

To keep this organized:

1. **Weekly**: Sort new downloads into proper folders
2. **Monthly**: Review and archive completed projects
3. **Quarterly**: Check for new duplicates
4. **Yearly**: Archive old files

## Quick Commands for You

```powershell
# Find files modified this week
Get-ChildItem -Recurse -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) }

# Find duplicates by name
Get-ChildItem -Recurse -File | Group-Object Name | Where-Object { $_.Count -gt 1 }

# Clean temp files (conversation ends)
Remove-Item -Recurse -Force "D:\cursor\file\.claude-temp\*"
```

Want to organize another folder?
```

## Examples

### Example 1: Organizing Downloads

**User**: "My Downloads folder is a mess with 500+ files. Help me organize it."

**Process**:
1. Analyzes Downloads folder
2. Finds patterns: work docs, personal photos, installers, random PDFs
3. Proposes structure:
   - Projects/ → Move to respective project directories
   - `Si Yuan\claude\` → Move non-project docs here
   - `.claude-temp\` → Temporary files
4. Asks for confirmation
5. Moves files intelligently based on content and names
6. Results: 500 files → organized across proper directories

### Example 2: Finding and Removing Duplicates

**User**: "Find duplicate files in my Documents and help me decide which to keep."

**Output**:
```markdown
# Found 23 Sets of Duplicates (156 MB total)

## Duplicate Set 1: "proposal.pdf"
- `D:\cursor\file\Documents\proposal.pdf` (2.3 MB, modified: 2026-01-15)
- `D:\cursor\file\Documents\old\proposal.pdf` (2.3 MB, modified: 2026-01-15)
- `D:\cursor\file\Desktop\proposal.pdf` (2.3 MB, modified: 2026-01-10)

**Recommendation**: Keep `Documents\proposal.pdf` (most recent in correct location)
Delete the other 2 copies?

[Continue for all duplicates...]
```

### Example 3: Organizing Non-Project Documents

**User**: "I have markdown files scattered everywhere. Organize them."

**Output**: Creates structure like:
```
D:\cursor\file\Si Yuan\claude\
├── 科普/
│   └── 低波投资策略_科普_2025-01-13.md
├── 分析/
│   └── 阿根廷现状_分析_2026-01-12.md
└── 笔记/
    └── Claude Code使用_笔记_2026-01-13.md
```

Then renames files following `主题_类型_日期.md` pattern.

## Common Organization Tasks

### Downloads Cleanup
```
Organize my Downloads folder - move non-project documents to Si Yuan\claude\,
project files to their respective directories, keep installers separate,
and archive files older than 3 months.
```

### Project Organization
```
Review my Projects folder structure and help me separate active
projects from old ones I should archive.
```

### Duplicate Removal
```
Find all duplicate files in my Documents folder and help me
decide which ones to keep.
```

### Desktop Cleanup
```
My Desktop is covered in files. Help me organize everything into
proper folders following my file organization rules.
```

### Document Organization
```
Organize all markdown files in Si Yuan\claude\ by type
(科普, 分析, 笔记) and rename them following the pattern.
```

## Pro Tips

1. **Start Small**: Begin with one messy folder (like Downloads) to build trust
2. **Regular Maintenance**: Run weekly cleanup on Downloads
3. **Consistent Naming**: Use "主题_类型_YYYY-MM-DD.md" format
4. **Archive Aggressively**: Move old projects to Archive instead of deleting
5. **Keep Active Separate**: Maintain clear boundaries between active and archived work
6. **Follow Your Rules**: Always use `Si Yuan\claude\` for non-project docs

## Best Practices

### Folder Naming
- Use clear, descriptive names
- Avoid spaces (use hyphens or underscores)
- Be specific: "client-proposals" not "docs"
- Use prefixes for ordering: "01-current", "02-archive"

### File Naming
- Include dates: "2026-01-13-meeting-notes.md"
- Be descriptive: "q3-financial-report.xlsx"
- Use Chinese names for better recognition
- Remove download artifacts: "document-final-v2 (1).pdf" → "document.pdf"
- Follow pattern: "主题_类型_日期.md"

### When to Archive
- Projects not touched in 6+ months
- Completed work that might be referenced later
- Old versions after migration to new systems
- Files you're hesitant to delete (archive first)

## Error Handling

**Important**: Always confirm before:
- Deleting files
- Moving large numbers of files
- Overwriting existing files
- Modifying system folders

If uncertain:
1. Stop and ask the user
2. Explain the situation clearly
3. Provide options
4. Wait for confirmation

## Related Use Cases

- Setting up organization for a new computer
- Preparing files for backup/archiving
- Cleaning up before storage cleanup
- Organizing shared team folders
- Structuring new project directories
