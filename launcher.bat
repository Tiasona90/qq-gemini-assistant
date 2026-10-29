@echo off
chcp 65001 >nul
title QQ Gemini 智能助手启动器
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if exist ".venv\Scripts\python.exe" (
    start "" ".venv\Scripts\pythonw.exe" gui_launcher.py
) else (
    python gui_launcher.py
)
exit
