@echo off
chcp 65001 >nul
echo 正在唤醒子铠的大脑和躯体...

:: 启动 Flask 后端 (新开一个窗口)
start cmd /k "call D:\anaconda\Scripts\activate.bat pytorch_env && python BackEnd/simple.py"

echo 启动指令已发送！请等待网页自动打开...
