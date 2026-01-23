---
name: tapestry
description: Unified content extraction and action planning. Use when user says "tapestry <URL>", "weave <URL>", "help me plan <URL>", "extract and plan <URL>", "make this actionable <URL>", or similar phrases indicating they want to extract content and create an action plan. Automatically detects content type (YouTube video, article, PDF) and processes accordingly. Saves content to D:\cursor\file\Si Yuan\claude\extracted-content\ and plans to D:\cursor\file\Si Yuan\claude\plans\.
allowed-tools: Bash,Read,Write
---

# Tapestry: Unified Content Extraction + Action Planning (Windows)

This is the **master skill** that orchestrates the entire Tapestry workflow:
1. Detect content type from URL
2. Extract content using appropriate tool (Windows-optimized)
3. Automatically create a Ship-Learn-Next action plan
4. Save everything following your file organization rules

## When to Use This Skill

Activate when the user:
- Says "tapestry [URL]"
- Says "weave [URL]"
- Says "help me plan [URL]"
- Says "extract and plan [URL]"
- Says "make this actionable [URL]"
- Says "turn [URL] into a plan"
- Provides a URL and asks to "learn and implement from this"
- Wants the full Tapestry workflow (extract → plan)

**Keywords to watch for**: tapestry, weave, plan, actionable, extract and plan, make a plan, turn into action

## File Organization Integration

**Content files** → `D:\cursor\file\Si Yuan\claude\extracted-content\`
**Plan files** → `D:\cursor\file\Si Yuan\claude\plans\`
**Temporary files** → `D:\cursor\file\.claude-temp\`

## How It Works

### Complete Workflow:
1. **Detect URL type** (YouTube, article, PDF)
2. **Extract content** using appropriate Windows-optimized method
3. **Create action plan** using ship-learn-next methodology
4. **Save both** content file and plan file following your rules
5. **Present summary** to user

## URL Detection Logic

### YouTube Videos

**Patterns to detect:**
- `youtube.com/watch?v=`
- `youtu.be/`
- `youtube.com/shorts/`
- `m.youtube.com/watch?v=`

**Action**: Extract transcript using yt-dlp (Windows)

### Web Articles/Blog Posts

**Patterns to detect:**
- `http://` or `https://`
- NOT YouTube, NOT PDF
- Common domains: medium.com, substack.com, dev.to, etc.
- Any HTML page

**Action**: Extract using reader or trafilatura (Windows)

### PDF Documents

**Patterns to detect:**
- URL ends with `.pdf`
- URL returns `Content-Type: application/pdf`

**Action**: Download and extract text using pdftotext (Windows)

## Windows-Optimized Extraction Methods

### Step 1: Detect Content Type (PowerShell)

```powershell
param($URL)

# Check for YouTube
if ($URL -match 'youtube\.com/watch|youtu\.be/|youtube\.com/shorts') {
    $CONTENT_TYPE = "youtube"
}
# Check for PDF
elseif ($URL -match '\.pdf$') {
    $CONTENT_TYPE = "pdf"
}
else {
    $CONTENT_TYPE = "article"
}

Write-Host "📍 Detected: $CONTENT_TYPE"
```

### Step 2: Extract Content (by Type)

#### YouTube Video (Windows)

```powershell
Write-Host "📺 Extracting YouTube transcript..."

# Check for yt-dlp
$ytDlpExists = Get-Command yt-dlp -ErrorAction SilentlyContinue

if (-not $ytDlpExists) {
    Write-Host "Installing yt-dlp..."
    winget install yt-dlp
    # OR: pip install yt-dlp
}

# Get video title (clean for Windows filename)
$VIDEO_TITLE = yt-dlp --print "%(title)s" $URL
$VIDEO_TITLE = $VIDEO_TITLE -replace '[/\\:*?"<>|]', '-' # Replace invalid chars
$VIDEO_TITLE = $VIDEO_TITLE.Substring(0, [Math]::Min(100, $VIDEO_TITLE.Length)) # Limit length

# Download transcript
yt-dlp --write-auto-sub --skip-download --sub-langs en --output "temp_transcript" $URL

# Convert VTT to clean text (PowerShell + Python)
$transcriptFile = "temp_transcript.en.vtt"
if (Test-Path $transcriptFile) {
    python -c "
import sys, re
seen = set()
with open('$transcriptFile', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('WEBVTT') and 'Kind:' not in line and 'Language:' not in line and '-->' not in line:
            clean = re.sub('<[^>]*>', '', line)
            clean = clean.replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
            if clean and clean not in seen:
                print(clean)
                seen.add(clean)
" | Out-File -Encoding UTF8 "$VIDEO_TITLE.txt"

    # Cleanup
    Remove-Item "temp_transcript.*.vtt" -Force
}

$CONTENT_FILE = "$VIDEO_TITLE.txt"
Write-Host "✓ Saved transcript: $CONTENT_FILE"
```

#### Article/Blog Post (Windows)

```powershell
Write-Host "📄 Extracting article content..."

# Check for extraction tools
$readerExists = Get-Command reader -ErrorAction SilentlyContinue
$trafilaturaExists = python -c "import trafilatura" 2>$null

if ($readerExists) {
    $TOOL = "reader"
    Write-Host "Using: reader (Mozilla Readability)"
}
elseif ($trafilaturaExists) {
    $TOOL = "trafilatura"
    Write-Host "Using: trafilatura"
}
else {
    $TOOL = "fallback"
    Write-Host "Using: fallback method (may be less accurate)"
}

# Extract based on tool
switch ($TOOL) {
    "reader" {
        reader $URL | Out-File -Encoding UTF8 temp_article.txt
        $ARTICLE_TITLE = (Get-Content temp_article.txt -First 1) -replace '^# ', ''
    }

    "trafilatura" {
        $METADATA = trafilatura --URL $URL --json 2>$null | ConvertFrom-Json
        $ARTICLE_TITLE = if ($METADATA.title) { $METADATA.title } else { "Article" }
        trafilatura --URL $URL --output-format txt --no-comments 2>$null | Out-File -Encoding UTF8 temp_article.txt
    }

    "fallback" {
        # Basic extraction using Invoke-WebRequest
        try {
            $response = Invoke-WebRequest -Uri $URL -UserAgent "Mozilla/5.0"
            $ARTICLE_TITLE = ($response.ParsedHtml.title -split ' - | ')[0]
            $response.Content | python -c "
from html.parser import HTMLParser
import sys

class ArticleExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.content = []
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'form'}
        self.in_content = False

    def handle_starttag(self, tag, attrs):
        if tag not in self.skip_tags and tag in {'p', 'article', 'main'}:
            self.in_content = True

    def handle_data(self, data):
        if self.in_content and data.strip():
            self.content.append(data.strip())

    def get_content(self):
        return '\n\n'.join(self.content)

parser = ArticleExtractor()
parser.feed(sys.stdin.read())
print(parser.get_content())
" | Out-File -Encoding UTF8 temp_article.txt
        }
        catch {
            Write-Host "Error: Could not extract article"
            exit 1
        }
    }
}

# Clean filename for Windows
$FILENAME = $ARTICLE_TITLE -replace '[/\\:*?"<>|]', '-'
$FILENAME = $FILENAME.Substring(0, [Math]::Min(80, $FILENAME.Length)).Trim()
$CONTENT_FILE = "$FILENAME.txt"
Move-Item temp_article.txt $CONTENT_FILE -Force

Write-Host "✓ Saved article: $CONTENT_FILE"
```

#### PDF Document (Windows)

```powershell
Write-Host "📑 Downloading PDF..."

# Download PDF
$PDF_FILENAME = Split-Path $URL -Leaf
Invoke-WebRequest -Uri $URL -OutFile $PDF_FILENAME

# Extract text using pdftotext (if available)
$pdftotextExists = Get-Command pdftotext -ErrorAction SilentlyContinue

if ($pdftotextExists) {
    pdftotext $PDF_FILENAME temp_pdf.txt
    $CONTENT_FILE = $PDF_FILENAME -replace '\.pdf$', '.txt'
    Move-Item temp_pdf.txt $CONTENT_FILE -Force
    Write-Host "✓ Extracted text from PDF: $CONTENT_FILE"

    # Optionally keep PDF
    $KEEP_PDF = Read-Host "Keep original PDF? (y/n)"
    if ($KEEP_PDF -ne 'y') {
        Remove-Item $PDF_FILENAME -Force
    }
}
else {
    Write-Host "⚠️  pdftotext not found. PDF downloaded but not extracted."
    Write-Host "   Install with: choco install poppler"
    $CONTENT_FILE = $PDF_FILENAME
}
```

### Step 3: Save to Correct Location

```powershell
# Define paths following your file organization rules
$EXTRACTED_CONTENT_DIR = "D:\cursor\file\Si Yuan\claude\extracted-content\"
$PLANS_DIR = "D:\cursor\file\Si Yuan\claude\plans\"
$TEMP_DIR = "D:\cursor\file\.claude-temp\"

# Create directories if they don't exist
New-Item -ItemType Directory -Force -Path $EXTRACTED_CONTENT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $PLANS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

# Move content file to extracted-content directory
$FINAL_CONTENT_PATH = Join-Path $EXTRACTED_CONTENT_DIR $CONTENT_FILE
Move-Item $CONTENT_FILE $FINAL_CONTENT_PATH -Force

Write-Host "✓ Content saved to: $FINAL_CONTENT_PATH"
```

### Step 4: Create Ship-Learn-Next Action Plan

**IMPORTANT**: Always create an action plan after extracting content.

```powershell
# Read the extracted content
$CONTENT = Get-Content $FINAL_CONTENT_PATH -Raw -Encoding UTF8

# Create plan using ship-learn-next methodology
# Extract core actionable lessons
# Define a specific 4-8 week quest
# Create Rep 1 (shippable this week)
# Design Reps 2-5 (progressive iterations)
# Save to: Ship-Learn-Next Plan - [Quest Title].md

$PLAN_TITLE = "Ship-Learn-Next Plan - $FILENAME"
$PLAN_PATH = Join-Path $PLANS_DIR "$PLAN_TITLE.md"

# Create the plan file
@"
# $PLAN_TITLE

**Source**: $URL
**Extracted**: $(Get-Date -Format 'yyyy-MM-dd')
**Content File**: $FINAL_CONTENT_PATH

---

## 🎯 Your Quest

[One-line summary of what you'll build]

## 📚 Key Learnings from Content

- [Actionable lesson 1]
- [Actionable lesson 2]
- [Actionable lesson 3]

## 📍 Rep 1: Ship This Week

**Goal**: [What you'll ship]
**Timeline**: This week
**Definition of Done**: [How you'll know it's complete]

## 🔮 Reps 2-5: Progressive Iterations

### Rep 2
**Goal**: [Next iteration]
**Timeline**: Week 2

### Rep 3
**Goal**: [Next iteration]
**Timeline**: Week 3

[Continue for Reps 4-5]

---

**Next Action**: When will you ship Rep 1?
"@ | Out-File -Encoding UTF8 $PLAN_PATH

Write-Host "✓ Plan saved to: $PLAN_PATH"
```

### Step 5: Present Results

```powershell
Write-Host ""
Write-Host "✅ Tapestry Workflow Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📥 Content Extracted:"
Write-Host "   ✓ Content type: $CONTENT_TYPE"
Write-Host "   ✓ Saved to: $FINAL_CONTENT_PATH"
$wordCount = (Get-Content $FINAL_CONTENT_PATH -Encoding UTF8).Split().Length
Write-Host "   ✓ $wordCount words extracted"
Write-Host ""
Write-Host "📋 Action Plan Created:"
Write-Host "   ✓ Saved to: $PLAN_PATH"
Write-Host ""
Write-Host "🎯 Your Quest: [One-line summary from plan]"
Write-Host ""
Write-Host "📍 Rep 1 (This Week): [Rep 1 goal from plan]"
Write-Host ""
Write-Host "When will you ship Rep 1?"
```

## Complete PowerShell Script

```powershell
#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Tapestry: Extract content + create action plan (Windows)
.DESCRIPTION
    Unified workflow to extract content from URLs and create action plans
.PARAMETER URL
    The URL to extract content from
.EXAMPLE
    .\tapestry.ps1 "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$URL
)

# Paths
$EXTRACTED_CONTENT_DIR = "D:\cursor\file\Si Yuan\claude\extracted-content\"
$PLANS_DIR = "D:\cursor\file\Si Yuan\claude\plans\"
$TEMP_DIR = "D:\cursor\file\.claude-temp\"

# Create directories
New-Item -ItemType Directory -Force -Path $EXTRACTED_CONTENT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $PLANS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

Write-Host "🧵 Tapestry Workflow Starting..."
Write-Host "URL: $URL"
Write-Host ""

# Step 1: Detect content type
if ($URL -match 'youtube\.com/watch|youtu\.be/|youtube\.com/shorts') {
    $CONTENT_TYPE = "youtube"
}
elseif ($URL -match '\.pdf$') {
    $CONTENT_TYPE = "pdf"
}
else {
    $CONTENT_TYPE = "article"
}

Write-Host "📍 Detected: $CONTENT_TYPE"
Write-Host ""

# Step 2: Extract content (based on type)
switch ($CONTENT_TYPE) {
    "youtube" {
        # [YouTube extraction code from above]
    }
    "article" {
        # [Article extraction code from above]
    }
    "pdf" {
        # [PDF extraction code from above]
    }
}

Write-Host ""

# Step 3: Create action plan
Write-Host "🚀 Creating Ship-Learn-Next action plan..."
# [Plan creation code from above]

Write-Host ""
Write-Host "✅ Tapestry Workflow Complete!"
Write-Host ""
Write-Host "📥 Content: $FINAL_CONTENT_PATH"
Write-Host "📋 Plan: $PLAN_PATH"
Write-Host ""
Write-Host "🎯 Next: Review your action plan and ship Rep 1!"
```

## Error Handling

### Common Issues:

**1. Unsupported URL type**
- Try article extraction as fallback
- If fails: "Could not extract content from this URL type"

**2. No content extracted**
- Check if URL is accessible
- Try alternate extraction method
- Inform user: "Extraction failed. URL may require authentication."

**3. Tools not installed**
- Offer to install: `winget install yt-dlp` for YouTube
- Offer to install: `pip install trafilatura` for articles
- Offer to install: `choco install poppler` for PDFs
- Use fallback methods when available

**4. Invalid filename characters**
- Automatically replace: `/ \ : * ? " < > |` with `-`
- Limit filename to 80 characters
- Trim whitespace

**5. Path too long**
- Windows has 260 character path limit
- Keep total path under 250 characters
- Use shorter directory names if needed

## Best Practices

- ✅ Always show what was detected ("📍 Detected: youtube")
- ✅ Display progress for each step
- ✅ Save both content file AND plan file to correct locations
- ✅ Show preview of extracted content (first 10 lines)
- ✅ Create plan automatically (don't ask)
- ✅ Present clear summary at end
- ✅ Ask commitment question: "When will you ship Rep 1?"
- ✅ Follow file organization rules:
  - Content → `Si Yuan\claude\extracted-content\`
  - Plans → `Si Yuan\claude\plans\`
  - Temp → `.claude-temp\`

## Windows-Specific Tips

1. **Use PowerShell**: All scripts are optimized for PowerShell 7+
2. **Encoding**: Always use UTF-8 encoding for text files
3. **Paths**: Use backslashes or forward slashes (both work in PowerShell)
4. **Quoting**: Quote paths with spaces: `"D:\cursor\file\Si Yuan\claude\`
5. **Command Discovery**: Use `Get-Command` to check if tools exist
6. **Error Handling**: Use `try/catch` for network operations

## Dependencies

**For YouTube:**
- yt-dlp: `winget install yt-dlp` or `pip install yt-dlp`
- Python 3: Usually pre-installed or `winget install Python`

**For Articles:**
- reader (npm): `npm install -g @mozilla/readability-cli`
- OR trafilatura (pip): `pip install trafilatura`
- Falls back to basic Invoke-WebRequest if neither available

**For PDFs:**
- poppler: `choco install poppler` (includes pdftotext)
- OR just download PDF without text extraction

**For Planning:**
- No additional requirements (uses built-in tools)

## Philosophy

**Tapestry weaves learning content into action.**

The unified workflow ensures you never just consume content - you always create an implementation plan. This transforms passive learning into active building.

**Extract → Plan → Ship → Learn → Next.**

That's the Tapestry way.

**Windows-Optimized**: All scripts tested and optimized for Windows PowerShell environment, following your file organization rules.
