@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao v1.0.4 dependency installer
echo ===================================================
echo.
echo This script creates a private project environment in .venv.
echo If Python 3.10/3.11 is missing, it will bootstrap a local Python 3.11 runtime.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "tools\bootstrap_release.ps1" %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo Please check your network connection, then rerun install.bat.
    echo Default PyTorch variant is CPU. For CUDA 12.8, run in PowerShell before install:
    echo     $env:XINBAO_TORCH_VARIANT="cu128"
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo [OK] Dependencies are installed in .venv.
echo Next step: double-click download_models.bat, then start_xinbao_desktop.bat.
echo.
pause
