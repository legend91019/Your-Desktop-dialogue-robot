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
$PayloadRoot = Join-Path $ProjectRoot "dist\Xinbao"
if ($Clean -and (Test-Path -LiteralPath $PayloadRoot)) {
    Remove-Item -LiteralPath $PayloadRoot -Recurse -Force
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
    $basePrefix = Split-Path -Parent (Resolve-Path -LiteralPath $PythonExe)
    if (-not (Test-Path -LiteralPath (Join-Path $basePrefix "python.exe"))) { throw "Cannot locate CPython base runtime" }
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
$launcherIcon = Join-Path $ProjectRoot "packaging\assets\xinbao.ico"
$pyInstallerArgs = @("--noconfirm", "--clean", "--onefile", "--name", "Xinbao", "--distpath", $payload, "--workpath", (Join-Path $ProjectRoot "build\pyinstaller"), "--specpath", (Join-Path $ProjectRoot "build\pyinstaller"))
if (Test-Path -LiteralPath $launcherIcon) { $pyInstallerArgs += @("--icon", $launcherIcon) }
$pyInstallerArgs += (Join-Path $ProjectRoot "tools\xinbao_entry.py")
& $VenvPython -m PyInstaller @pyInstallerArgs
if ($LASTEXITCODE -ne 0) { throw "Failed to build Xinbao.exe launcher" }

Copy-Item -LiteralPath $RuntimePythonDir -Destination (Join-Path $payload "python") -Recurse -Force
Copy-Item -LiteralPath $VenvDir -Destination (Join-Path $payload ".venv") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "BackEnd") -Destination (Join-Path $payload "BackEnd") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "FrontEnd") -Destination (Join-Path $payload "FrontEnd") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "utils") -Destination (Join-Path $payload "utils") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets") -Destination (Join-Path $payload "assets") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "models") -Destination (Join-Path $payload "models") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "tools") -Destination (Join-Path $payload "tools") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "runtime_paths.py") -Destination (Join-Path $payload "runtime_paths.py") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "startup_checks.py") -Destination (Join-Path $payload "startup_checks.py") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "desktop_launcher.py") -Destination (Join-Path $payload "desktop_launcher.py") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "knowledge.md") -Destination (Join-Path $payload "knowledge.md") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "config.example.json") -Destination (Join-Path $payload "config.example.json") -Force

# Keep the installer payload focused on runtime files. Build-only tooling and
# bytecode/test caches can be regenerated and are not needed by end users.
$sitePackages = Join-Path $payload ".venv\Lib\site-packages"
$removeFromPayload = @(
    (Join-Path $payload "tools"),
    (Join-Path $sitePackages "PyInstaller"),
    (Join-Path $sitePackages "pip"),
    (Join-Path $sitePackages "setuptools"),
    (Join-Path $sitePackages "wheel"),
    (Join-Path $sitePackages "modelscope"),
    (Join-Path $sitePackages "kubernetes"),
    (Join-Path $sitePackages "build"),
    (Join-Path $sitePackages "pyproject_hooks"),
    (Join-Path $sitePackages "altgraph"),
    (Join-Path $sitePackages "pefile.py"),
    (Join-Path $sitePackages "peutils.py"),
    (Join-Path $sitePackages "torch\include"),
    (Join-Path $sitePackages "onnxruntime\transformers"),
    (Join-Path $payload "python\Lib\site-packages"),
    (Join-Path $payload "BackEnd\test.py"),
    (Join-Path $payload "utils\Classifier\test.py"),
    (Join-Path $payload "utils\Retriever\test.py"),
    (Join-Path $payload "utils\Retriever\input.docx")
)
foreach ($path in $removeFromPayload) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
Get-ChildItem -LiteralPath $payload -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "__pycache__" } |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $payload -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @(".pyc", ".pyo", ".whl") } |
    Remove-Item -Force
Get-ChildItem -LiteralPath $payload -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq ".idea" -or $_.Name -eq "._____temp" } |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force

Write-Host "Release runtime ready: $VenvDir"
