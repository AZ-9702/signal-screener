@echo off
chcp 65001 >nul
echo Signal Screener - Incremental Update (last 3 days)
echo.
python "%~dp0signal_screener.py" update
echo.
pause
