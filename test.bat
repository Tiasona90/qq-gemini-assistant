@echo off
chcp 65001 >nul
echo ========================================================
echo   运行自动化测试套件 (Pytest)
echo ========================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

".venv\Scripts\pytest.exe" tests -v
pause
