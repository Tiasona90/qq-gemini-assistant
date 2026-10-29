@echo off
chcp 65001 >nul
echo ========================================================
echo   正在编译 QQ Gemini 智能助手 Windows EXE 启动器
echo ========================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set CSC_PATH=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe

if not exist "%CSC_PATH%" (
    echo [错误] 未在系统中检测到 C# 编译器 csc.exe
    pause
    exit /b 1
)

echo 正在编译原生 WinGUI EXE 启动器 (QQGeminiAssistant.exe)...
"%CSC_PATH%" /target:winexe /win32icon:"assets\launcher.ico" /out:"QQGeminiAssistant.exe" "launcher.cs"

if %ERRORLEVEL% EQU 0 (
    echo [成功] 已生成原生启动器：QQGeminiAssistant.exe
) else (
    echo [失败] 编译出现错误。
)

pause
