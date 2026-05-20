@echo off
:: 设置编码为UTF-8，防止终端中文乱码
chcp 65001 >nul

echo ===================================================
echo   IntelliChat 一键下载模型与分类器训练脚本
echo ===================================================
echo.

:: 1. 执行下载所有模型的脚本
echo [步骤 1/2] 正在运行 download_all_models.py 下载所需模型...
python download_all_models.py
if %errorlevel% neq 0 (
    echo.
    echo [错误] 模型下载失败，请检查网络连接或Python环境！
    goto error
)
echo [成功] 所有模型下载完成！
echo.

:: 2. 执行训练分类器的脚本
echo [步骤 2/2] 正在运行 train_classifier.py 开始训练分类器...
python train_classifier.py
if %errorlevel% neq 0 (
    echo.
    echo [错误] 分类器训练失败，请检查数据集或脚本代码！
    goto error
)
echo [成功] 分类器训练完成！
echo.

echo ===================================================
echo   所有任务已成功执行完毕！
echo ===================================================
goto end

:error
echo.
echo 程序因错误中止。
echo ===================================================

:end
pause