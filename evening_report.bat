@echo off
REM Alpha-Prime evening validation (run after 15:35 IST on every trading day of the experiment).
REM   Track A: today's paper trades - P&L vs NIFTY, slippage, charges, risk-limit checks
REM   Tracks B/C: frozen index_trend + baseline replayed on today's fresh 5-min bars
cd /d "%~dp0"
venv\Scripts\python scripts\daily_report.py
venv\Scripts\python research\strat_lab\forward_test.py
pause
