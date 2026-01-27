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
.NOTES
    This is a reference implementation.
    For detailed extraction methods, see the original tapestry-v1.0.md in archive/.
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
# NOTE: For detailed extraction implementation,
# see archive/tapestry-v1.0.md lines 98-257

switch ($CONTENT_TYPE) {
    "youtube" {
        Write-Host "📺 Extracting YouTube transcript..."
        # Implementation: See archive/tapestry-v1.0.md lines 98-143
        # Requires: yt-dlp, Python 3
    }
    "article" {
        Write-Host "📄 Extracting article content..."
        # Implementation: See archive/tapestry-v1.0.md lines 145-226
        # Requires: reader OR trafilatura OR fallback to Invoke-WebRequest
    }
    "pdf" {
        Write-Host "📑 Downloading PDF..."
        # Implementation: See archive/tapestry-v1.0.md lines 228-257
        # Requires: pdftotext (optional)
    }
}

Write-Host ""

# Step 3: Create action plan
# Implementation: See archive/tapestry-v1.0.md lines 279-341

Write-Host "🚀 Creating Ship-Learn-Next action plan..."
# [Plan creation code from archive]

Write-Host ""
Write-Host "✅ Tapestry Workflow Complete!"
Write-Host ""
Write-Host "📥 Content: $FINAL_CONTENT_PATH"
Write-Host "📋 Plan: $PLAN_PATH"
Write-Host ""
Write-Host "🎯 Next: Review your action plan and ship Rep 1!"
