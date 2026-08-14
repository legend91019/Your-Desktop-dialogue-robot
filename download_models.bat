@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao model downloader
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

"%PYTHON_EXE%" download_all_models.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Model download failed. Please check network connection or mirror settings.
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo [OK] Base models are ready.
echo.
if not exist "assets\classifier\route_classifier.joblib" (
    echo [ERROR] Packaged route classifier is missing:
    echo     assets\classifier\route_classifier.joblib
    echo Please download a complete release package.
    echo.
    pause
    exit /b 1
)

echo [OK] Packaged route classifier is ready: assets\classifier\route_classifier.joblib
echo Optional Index-TTS models are not downloaded by default.
echo To download them later, run:
echo     "%PYTHON_EXE%" download_all_models.py --with-indextts
echo.
pause
