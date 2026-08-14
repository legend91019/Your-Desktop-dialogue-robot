@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao model download
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

echo [1/2] Running download_all_models.py...
"%PYTHON_EXE%" download_all_models.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Model download failed. Please check network connection.
    goto error
)
echo [OK] Base models are ready.
echo.

echo [2/2] Checking packaged route classifier...
if not exist "assets\classifier\route_classifier.joblib" (
    echo [ERROR] Packaged route classifier is missing:
    echo     assets\classifier\route_classifier.joblib
    goto error
)
echo [OK] Packaged route classifier is ready.
echo.

echo ===================================================
echo   All tasks finished.
echo ===================================================
goto end

:error
echo.
echo The script stopped because of an error.
echo ===================================================

:end
pause
