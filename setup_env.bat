@echo off
title Setup Python Virtual Environment for KKBOX Project
echo ======================================================================
echo Setting up Python Virtual Environment (.venv)...
echo ======================================================================
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.11, 3.12, or 3.13.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment in .venv...
    python -m venv .venv
) else (
    echo Virtual environment .venv already exists.
)

echo Activating .venv and installing requirements...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ======================================================================
echo Setup completed successfully!
echo You can now open this folder in VS Code.
echo ======================================================================
pause
