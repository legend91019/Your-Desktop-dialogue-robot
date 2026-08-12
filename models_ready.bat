@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Xinbao basic model and classifier setup
echo ===================================================
echo.

echo [1/2] Downloading basic embedding and reranker models...
python download_all_models.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Basic model download failed.
    echo Please check the network connection and Python environment.
    goto error
)
echo [OK] Basic models are ready.
echo.

echo [2/2] Training the local classifier...
python train_classifier.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Classifier training failed.
    echo Please check classifier_corpus.csv and the Python dependencies.
    goto error
)
echo [OK] Classifier training completed.
echo.

echo ===================================================
echo   Setup completed.
echo   Index-TTS is optional and was not downloaded.
echo   To download it later, run:
echo     python download_all_models.py --with-indextts
echo ===================================================
goto end

:error
echo.
echo Setup stopped because of an error.
pause
exit /b 1

:end
pause
