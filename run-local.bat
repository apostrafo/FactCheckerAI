@echo off
setlocal

REM This script helps set up and run the application locally

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PYTHON_CMD=py
    ) else (
        echo Error: Python is not installed or not in your PATH.
        echo Please install Python 3.9+ before continuing.
        exit /b 1
    )
)

REM Check if pip is available
where pip >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PIP_CMD=pip
) else (
    echo Error: pip is not installed or not in your PATH.
    echo Please make sure pip is installed with your Python.
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Creating .env file...
    set /p api_key=Enter your OpenRouter API key: 
    echo OPENROUTER_API_KEY=%api_key%> .env
    echo .env file created successfully.
) else (
    echo .env file already exists.
)

REM Check if virtual environment exists
if not exist venv\ (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the application
echo Starting FactCheckerAI...
python app.py

REM Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat
exit /b 0 