@echo off
chcp 65001 >nul
echo 正在停止 QQ Gemini 智能助手进程...
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*qq_gemini_assistant*main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host 'Stopped process' $_.ProcessId }"
echo 停止完成。
