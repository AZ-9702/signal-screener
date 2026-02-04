@echo off
chcp 65001 >nul
echo ========================================
echo   Signal Screener - Local Server
echo ========================================
echo.
echo Starting server with auto-update...
echo Press Ctrl+C to stop, or close this window.
echo.
python "%~dp0signal_screener.py" serve --update
echo.
echo ========================================
echo Server stopped or error occurred.
echo ========================================
pause
