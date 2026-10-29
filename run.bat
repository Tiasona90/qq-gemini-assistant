@echo off
chcp 65001 >nul
echo ========================================================
echo   启动 QQ Gemini 智能助手服务 (OneBot 11 服务端)
echo ========================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv，请先运行初始化命令。
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py %*
