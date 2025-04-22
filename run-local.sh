#!/bin/bash

# This script helps set up and run the application locally

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python is not installed or not in your PATH."
    echo "Please install Python 3.9+ before continuing."
    exit 1
fi

# Check if pip is available
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "Error: pip is not installed or not in your PATH."
    echo "Please make sure pip is installed with your Python."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    read -p "Enter your OpenRouter API key: " api_key
    echo "OPENROUTER_API_KEY=$api_key" > .env
    echo ".env file created successfully."
else
    echo ".env file already exists."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
$PIP_CMD install -r requirements.txt

# Run the application
echo "Starting FactCheckerAI..."
python app.py 