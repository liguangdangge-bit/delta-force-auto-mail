$ErrorActionPreference = 'Stop'
$MailPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $MailPython)) { throw 'Run setup_environment.ps1 first.' }
Push-Location -LiteralPath $PSScriptRoot
try { & $MailPython 'main.py' }
finally { Pop-Location }
