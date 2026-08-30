param(
    [string]$Version = "1.0.0",
    [string]$IsccPath = "ISCC.exe",
    [string]$PythonExe = "python",
    [switch]$SkipRuntimeBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $ProjectRoot "dist"
$PayloadDir = Join-Path $DistDir "Xinbao"
$InstallerDir = Join-Path $DistDir "installer"

Set-Location -LiteralPath $ProjectRoot
if (-not $SkipRuntimeBuild) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "tools\build_runtime.ps1") -Clean -PythonExe $PythonExe
    if ($LASTEXITCODE -ne 0) { throw "Runtime build failed" }
}

if (-not (Test-Path -LiteralPath $PayloadDir)) {
    throw "Missing installer payload: $PayloadDir"
}
if (-not (Test-Path -LiteralPath (Join-Path $PayloadDir "Xinbao.exe"))) {
    throw "Missing Xinbao.exe in installer payload"
}

# Always clean the payload, including when -SkipRuntimeBuild is used after tests.
$payloadSitePackages = Join-Path $PayloadDir ".venv\Lib\site-packages"
$payloadCleanup = @(
    (Join-Path $PayloadDir "tools"),
    (Join-Path $PayloadDir "python\Lib\site-packages"),
    (Join-Path $PayloadDir "BackEnd\test.py"),
    (Join-Path $PayloadDir "utils\Classifier\test.py"),
    (Join-Path $PayloadDir "utils\Retriever\test.py"),
    (Join-Path $PayloadDir "utils\Retriever\input.docx"),
    (Join-Path $payloadSitePackages "PyInstaller"),
    (Join-Path $payloadSitePackages "pip"),
    (Join-Path $payloadSitePackages "setuptools"),
    (Join-Path $payloadSitePackages "wheel"),
    (Join-Path $payloadSitePackages "modelscope"),
    (Join-Path $payloadSitePackages "kubernetes"),
    (Join-Path $payloadSitePackages "build"),
    (Join-Path $payloadSitePackages "pyproject_hooks"),
    (Join-Path $payloadSitePackages "altgraph"),
    (Join-Path $payloadSitePackages "pefile.py"),
    (Join-Path $payloadSitePackages "peutils.py"),
    (Join-Path $payloadSitePackages "torch\include"),
    (Join-Path $payloadSitePackages "onnxruntime\transformers")
)
foreach ($path in $payloadCleanup) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
}
Get-ChildItem -LiteralPath $PayloadDir -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "__pycache__" -or $_.Name -eq ".idea" -or $_.Name -eq "._____temp" } |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $PayloadDir -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @(".pyc", ".pyo", ".whl") } |
    Remove-Item -Force

New-Item -ItemType Directory -Force -Path $InstallerDir | Out-Null
$iss = Join-Path $ProjectRoot "packaging\Xinbao.iss"
& $IsccPath "/DAppVersion=$Version" $iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed" }

$installer = Join-Path $InstallerDir "Xinbao-Setup-v$Version.exe"
if (-not (Test-Path -LiteralPath $installer)) { throw "Installer was not generated: $installer" }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$installer.sha256" -Value "$hash  $(Split-Path -Leaf $installer)" -Encoding ascii
Write-Host "Installer ready: $installer"
