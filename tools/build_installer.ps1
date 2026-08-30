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

New-Item -ItemType Directory -Force -Path $InstallerDir | Out-Null
$iss = Join-Path $ProjectRoot "packaging\Xinbao.iss"
& $IsccPath "/DAppVersion=$Version" $iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed" }

$installer = Join-Path $InstallerDir "Xinbao-Setup-v$Version.exe"
if (-not (Test-Path -LiteralPath $installer)) { throw "Installer was not generated: $installer" }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$installer.sha256" -Value "$hash  $(Split-Path -Leaf $installer)" -Encoding ascii
Write-Host "Installer ready: $installer"
