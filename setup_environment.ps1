$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt 'pyinstaller==6.21.0'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
finally { Pop-Location }
