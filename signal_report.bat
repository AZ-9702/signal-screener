@echo off
chcp 65001 >nul
echo Signal Screener - Generating Master Report...
echo.
python "%~dp0signal_screener.py" report
echo.
pause
