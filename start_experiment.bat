@echo off
REM Alpha-Prime paper-trading day: pre-flight check, then start the scheduler.
REM Leave this window open until after 15:30. Press Ctrl+C to stop.
cd /d "%~dp0"
venv\Scripts\python scripts\preflight_check.py
if errorlevel 1 (
    echo Pre-flight failed - not starting. Fix the FAIL lines above.
    pause
    exit /b 1
)
venv\Scripts\python run_trading.py
pause
