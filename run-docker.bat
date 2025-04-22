@echo off
setlocal

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Docker is not installed. Please install Docker first.
    exit /b 1
)

REM Check if docker-compose is installed
where docker-compose >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Docker Compose is not installed. Please install Docker Compose first.
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

REM Build and start the Docker container
echo Building and starting the Docker container...
docker-compose up --build

REM Exit gracefully when docker-compose is terminated
exit /b 0 