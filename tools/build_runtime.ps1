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
$RuntimePythonDir = Join-Path $BuildRoot "python"

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

if (-not (Test-Path -LiteralPath (Join-Path $RuntimePythonDir "python.exe"))) {
    $basePrefix = & $PythonExe -c "import sys; print(sys.base_prefix)"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $basePrefix)) { throw "Cannot locate CPython base runtime" }
    Copy-Item -LiteralPath $basePrefix -Destination $RuntimePythonDir -Recurse -Force
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install torch torchvision torchaudio --index-url $TorchIndexUrl
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install release dependencies" }
& $VenvPython -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) { throw "Failed to install build launcher dependency" }

if (-not $SkipModels) {
    & $VenvPython (Join-Path $ProjectRoot "download_all_models.py")
    if ($LASTEXITCODE -ne 0) { throw "Failed to download release models" }
}

& $VenvPython (Join-Path $ProjectRoot "tools\create_model_manifest.py")
if ($LASTEXITCODE -ne 0) { throw "Failed to create model manifest" }

$payload = Join-Path $ProjectRoot "dist\Xinbao"
New-Item -ItemType Directory -Force -Path $payload | Out-Null
& $VenvPython -m PyInstaller --noconfirm --clean --onefile --name Xinbao --distpath $payload --workpath (Join-Path $ProjectRoot "build\pyinstaller") --specpath (Join-Path $ProjectRoot "build\pyinstaller") (Join-Path $ProjectRoot "tools\xinbao_entry.py")
if ($LASTEXITCODE -ne 0) { throw "Failed to build Xinbao.exe launcher" }

Copy-Item -LiteralPath $RuntimePythonDir -Destination (Join-Path $payload "python") -Recurse -Force
Copy-Item -LiteralPath $VenvDir -Destination (Join-Path $payload ".venv") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "BackEnd") -Destination (Join-Path $payload "BackEnd") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "FrontEnd") -Destination (Join-Path $payload "FrontEnd") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "utils") -Destination (Join-Path $payload "utils") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets") -Destination (Join-Path $payload "assets") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "models") -Destination (Join-Path $payload "models") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "tools") -Destination (Join-Path $payload "tools") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "knowledge.md") -Destination (Join-Path $payload "knowledge.md") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "config.example.json") -Destination (Join-Path $payload "config.example.json") -Force

Write-Host "Release runtime ready: $VenvDir"
