param(
    [switch]$CreateVenvOnly
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$LocalPythonDir = Join-Path $ProjectRoot ".python"
$ToolsDir = Join-Path $ProjectRoot ".tools"
$UvDir = Join-Path $ToolsDir "uv"
$UvZip = Join-Path $UvDir "uv.zip"
$UvDownloadUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

function Get-PythonVersion {
    param([string[]]$Command)

    try {
        $exe = $Command[0]
        $cmdArgs = @()
        if ($Command.Length -gt 1) {
            $cmdArgs = $Command[1..($Command.Length - 1)]
        }

        $output = & $exe @cmdArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $output) {
            return $null
        }
        return [version]($output | Select-Object -Last 1)
    }
    catch {
        return $null
    }
}

function Test-SupportedPython {
    param([version]$Version)

    if ($null -eq $Version) {
        return $false
    }
    return ($Version.Major -eq 3 -and ($Version.Minor -eq 10 -or $Version.Minor -eq 11))
}

function Find-CompatiblePython {
    $candidates = @()

    if ($env:XINBAO_PYTHON) {
        $candidates += ,@($env:XINBAO_PYTHON)
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += ,@("py", "-3.11")
        $candidates += ,@("py", "-3.10")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += ,@("python")
    }

    foreach ($candidate in $candidates) {
        $version = Get-PythonVersion -Command $candidate
        if (Test-SupportedPython -Version $version) {
            Write-Host "Using Python $version from: $($candidate -join ' ')"
            return $candidate
        }
    }

    return $null
}

function Get-UvExe {
    $existing = Get-ChildItem -LiteralPath $UvDir -Recurse -Filter "uv.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing) {
        return $existing.FullName
    }

    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    Write-Host "Downloading uv bootstrapper..."
    Invoke-WebRequest -Uri $UvDownloadUrl -OutFile $UvZip
    Expand-Archive -LiteralPath $UvZip -DestinationPath $UvDir -Force

    $downloaded = Get-ChildItem -LiteralPath $UvDir -Recurse -Filter "uv.exe" | Select-Object -First 1
    if (-not $downloaded) {
        throw "uv.exe was not found after extracting $UvZip"
    }

    return $downloaded.FullName
}

function Ensure-Venv {
    if (Test-Path -LiteralPath $VenvPython) {
        Write-Host "Reusing existing project environment: $VenvPython"
        return
    }

    $python = Find-CompatiblePython
    if ($python) {
        Write-Host "Creating .venv with compatible system Python..."
        $pythonExe = $python[0]
        $pythonArgs = @()
        if ($python.Length -gt 1) {
            $pythonArgs = $python[1..($python.Length - 1)]
        }

        & $pythonExe @pythonArgs -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create .venv with $($python -join ' ')"
        }
        return
    }

    Write-Host "No compatible Python 3.10/3.11 was found."
    Write-Host "Creating project-local Python 3.11 under .python by uv venv..."

    $uv = Get-UvExe
    $env:UV_PYTHON_INSTALL_DIR = $LocalPythonDir
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

    & $uv venv $VenvDir --python 3.11 --python-preference managed
    if ($LASTEXITCODE -ne 0) {
        throw "uv failed to create .venv with managed Python 3.11"
    }
}

Set-Location -LiteralPath $ProjectRoot
Ensure-Venv

if ($CreateVenvOnly) {
    exit 0
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw ".venv was not created successfully: $VenvPython"
}

& $VenvPython (Join-Path $ProjectRoot "tools\setup_env.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
