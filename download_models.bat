@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao model downloader and classifier trainer
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
echo [2/2] Training local classifier...
"%PYTHON_EXE%" train_classifier.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Classifier training failed. Please check classifier_corpus.csv and dependencies.
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo [OK] Classifier training is complete.
echo Optional Index-TTS models are not downloaded by default.
echo To download them later, run:
echo     "%PYTHON_EXE%" download_all_models.py --with-indextts
echo.
pause
