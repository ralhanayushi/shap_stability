@echo off
title KKBOX SHAP Stability Pipeline Run
echo ======================================================================
echo Executing End-to-End SHAP Explanation Stability Pipeline...
echo ======================================================================
cd /d "%~dp0"
python run_pipeline.py --data-dir data/raw --output-dir results --max-users 25000
pause
