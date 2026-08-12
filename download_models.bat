@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao v1.0.0 basic model downloader
echo ===================================================
echo.

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"

echo Downloading basic embedding and reranker models...
"%PYTHON_EXE%" download_all_models.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Basic model download failed.
    echo Please check the network connection and Python environment.
    pause
    exit /b 1
)

echo.
echo Basic models are ready.
echo Index-TTS is optional. To download it later, run:
echo   "%PYTHON_EXE%" download_all_models.py --with-indextts
echo.
pause
