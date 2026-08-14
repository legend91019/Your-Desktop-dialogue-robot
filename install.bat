@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao v1.0.0 dependency installer
echo ===================================================
echo.
echo This script creates a private project environment in .venv.
echo Recommended Python: 3.10 or 3.11.
echo.

set "BOOTSTRAP_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "BOOTSTRAP_PYTHON=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" (
    where python >nul 2>nul
    if errorlevel 1 set "BOOTSTRAP_PYTHON=py"
)

"%BOOTSTRAP_PYTHON%" tools\setup_env.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo If your PC has no Python 3.10/3.11, install one and rerun this file.
    echo For CPU-only installation, run:
    echo     python tools\setup_env.py --torch cpu
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo [OK] Dependencies are installed in .venv.
echo Next step: double-click download_models.bat, then start_xinbao_desktop.bat.
echo.
pause
