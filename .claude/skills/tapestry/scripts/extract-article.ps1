<#
.SYNOPSIS
    Extract article/blog post content
.DESCRIPTION
    Extracts readable content from web articles
.PARAMETER URL
    Article URL
.EXAMPLE
    .\extract-article.ps1 "https://example.com/article"
.NOTES
    Full implementation in archive/tapestry-v1.0.md lines 145-226
    Requires: reader OR trafilatura (optional, has fallback)
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$URL
)

Write-Host "📄 Extracting article content..."

# Check for extraction tools
$readerExists = Get-Command reader -ErrorAction SilentlyContinue
try {
    $null = python -c "import trafilatura" 2>$null
    $trafilaturaExists = $true
} catch {
    $trafilaturaExists = $false
}

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
        try {
            $response = Invoke-WebRequest -Uri $URL -UserAgent "Mozilla/5.0"
            $ARTICLE_TITLE = ($response.ParsedHtml.title -split ' - | ')[0]
            $response.Content | python -c @"
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
"@ | Out-File -Encoding UTF8 temp_article.txt
        }
        catch {
            Write-Host "❌ Error: Could not extract article"
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
