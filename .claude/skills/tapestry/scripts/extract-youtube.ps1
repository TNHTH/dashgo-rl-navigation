<#
.SYNOPSIS
    Extract YouTube video transcript
.DESCRIPTION
    Downloads transcript using yt-dlp and converts to clean text
.PARAMETER URL
    YouTube video URL
.EXAMPLE
    .\extract-youtube.ps1 "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
.NOTES
    Full implementation in archive/tapestry-v1.0.md lines 98-143
    Requires: yt-dlp (`winget install yt-dlp`)
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$URL
)

# Check for yt-dlp
$ytDlpExists = Get-Command yt-dlp -ErrorAction SilentlyContinue

if (-not $ytDlpExists) {
    Write-Host "❌ yt-dlp not found. Install with: winget install yt-dlp"
    exit 1
}

# Get video title (clean for Windows filename)
$VIDEO_TITLE = yt-dlp --print "%(title)s" $URL
$VIDEO_TITLE = $VIDEO_TITLE -replace '[/\\:*?"<>|]', '-'
$VIDEO_TITLE = $VIDEO_TITLE.Substring(0, [Math]::Min(100, $VIDEO_TITLE.Length))

Write-Host "📺 Extracting: $VIDEO_TITLE"

# Download transcript
yt-dlp --write-auto-sub --skip-download --sub-langs en --output "temp_transcript" $URL

# Convert VTT to clean text
$transcriptFile = "temp_transcript.en.vtt"
if (Test-Path $transcriptFile) {
    python -c @"
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
"@ | Out-File -Encoding UTF8 "$VIDEO_TITLE.txt"

    # Cleanup
    Remove-Item "temp_transcript.*.vtt" -Force
    Write-Host "✓ Saved: $VIDEO_TITLE.txt"
} else {
    Write-Host "❌ No transcript found for this video"
}
