param(
    [string]$PythonExe = "python",
    [switch]$Clean,
    [switch]$SkipModels,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $ProjectRoot "build\runtime"
$VenvDir = Join-Path $BuildRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot
if ($Clean -and (Test-Path -LiteralPath $BuildRoot)) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

$version = & $PythonExe --version 2>&1
if ($LASTEXITCODE -ne 0 -or $version -notmatch "Python 3\.11\.") {
    throw "The release runtime must be built with CPython 3.11.x. Detected: $version"
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create release venv at $VenvDir" }
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install torch torchvision torchaudio --index-url $TorchIndexUrl
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install release dependencies" }

if (-not $SkipModels) {
    & $VenvPython (Join-Path $ProjectRoot "download_all_models.py")
    if ($LASTEXITCODE -ne 0) { throw "Failed to download release models" }
}

& $VenvPython (Join-Path $ProjectRoot "tools\create_model_manifest.py")
if ($LASTEXITCODE -ne 0) { throw "Failed to create model manifest" }

Write-Host "Release runtime ready: $VenvDir"
