@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao v1.0.0 dependency installer
echo ===================================================
echo.

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"

echo [1/2] Checking Python...
"%PYTHON_EXE%" --version
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python was not found.
    echo Please install Python 3.10+ or activate your conda environment first.
    pause
    exit /b 1
)

echo.
echo [2/2] Installing project dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo You can retry after switching PyPI mirror or checking your Python environment.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   Install completed.
echo   Next:
echo     download_models.bat
echo     start_xinbao_desktop.bat
echo ===================================================
pause
