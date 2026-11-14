@echo off
title KKBOX SHAP Stability Evaluation Dashboard
echo ======================================================================
echo Launching KKBOX SHAP Stability Research Dashboard...
echo ======================================================================
cd /d "%~dp0"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
pause
