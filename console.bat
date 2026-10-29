@echo off
chcp 65001 >nul
echo ========================================================
echo   启动控制台交互模拟终端 (无需 QQ 客户端，直接测试对话与指令)
echo ========================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

".venv\Scripts\python.exe" main.py --console %*
