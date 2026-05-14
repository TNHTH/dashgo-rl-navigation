<#
.SYNOPSIS
    Download and extract PDF content
.DESCRIPTION
    Downloads PDF and extracts text using pdftotext
.PARAMETER URL
    PDF URL
.EXAMPLE
    .\extract-pdf.ps1 "https://example.com/document.pdf"
.NOTES
    Full implementation in archive/tapestry-v1.0.md lines 228-257
    Requires: pdftotext from poppler (`choco install poppler`)
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$URL
)

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
