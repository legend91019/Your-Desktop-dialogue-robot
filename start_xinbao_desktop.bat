@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao Desktop Launcher
echo ===================================================
echo.

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] .venv was not found.
    echo Please run install.bat first.
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" desktop_launcher.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Xinbao desktop failed to start.
    echo If pywebview is missing, run:
    echo     "%PYTHON_EXE%" -m pip install pywebview
    echo Or rerun install.bat.
    echo.
    pause
    exit /b %errorlevel%
)
