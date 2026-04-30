@echo off
chcp 65001 >nul
echo 正在唤醒子铠的大脑和躯体...

:: 启动 Flask 后端 (新开一个窗口)
start cmd /k "call D:\anaconda\Scripts\activate.bat pytorch_env && python BackEnd/simple.py"

:: 等待 12 秒，确保后端先启动
timeout /t 5

:: 启动 Vue 前端 (新开一个窗口)
start cmd /k "cd FrontEnd && npm run dev"

echo 启动指令已发送！请等待网页自动打开...
:: 自动打开浏览器
start http://localhost:5173