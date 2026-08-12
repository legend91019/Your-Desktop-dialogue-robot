@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao Desktop Launcher
echo ===================================================
echo.

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"

"%PYTHON_EXE%" desktop_launcher.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Xinbao desktop launcher failed.
    echo If pywebview is missing, run:
    echo     "%PYTHON_EXE%" -m pip install pywebview
    echo Or reinstall all dependencies:
    echo     "%PYTHON_EXE%" -m pip install -r requirements.txt
    echo.
    pause
    exit /b %errorlevel%
)
