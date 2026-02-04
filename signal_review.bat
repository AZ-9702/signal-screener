@echo off
chcp 65001 >nul
echo Signal Screener - Review Mode (last 15 days filings)
echo.
python "%~dp0signal_screener.py" update --days 15 --review
echo.
pause
