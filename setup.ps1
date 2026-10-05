# Podesavanje okruzenja za NorthStar RAG (Windows)
# Pokretanje iz foldera projekta:
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== 1/4 Provera Pythona 3.12 ==" -ForegroundColor Cyan
$pyOk = $false
try { py -3.12 --version; if ($LASTEXITCODE -eq 0) { $pyOk = $true } } catch { }
if (-not $pyOk) {
    Write-Host "Python 3.12 nije pronadjen." -ForegroundColor Red
    Write-Host "Instalirajte ga komandom:  winget install Python.Python.3.12"
    Write-Host "ili sa https://www.python.org/downloads/ (oznacite 'Add python.exe to PATH')."
    Write-Host "Zatim zatvorite i ponovo otvorite VS Code i pokrenite ovu skriptu ponovo."
    exit 1
}

Write-Host "== 2/4 Virtuelno okruzenje .venv ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "== 3/4 Instalacija paketa (prvi put traje nekoliko minuta) ==" -ForegroundColor Cyan
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Instalacija paketa nije uspela - pogledajte gresku iznad." -ForegroundColor Red; exit 1 }

Write-Host "== 4/4 .env i Git ==" -ForegroundColor Cyan
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Napravljen .env - upisite API kljuc u njega." }
if (-not (Test-Path ".git")) {
    git init -b main
    Write-Host "Git repozitorijum inicijalizovan (grana main)."
}

Write-Host ""
Write-Host "Gotovo. U VS Code-u: Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv" -ForegroundColor Green
