#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose is not installed. Please install Docker Compose first."
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

# Build and start the Docker container
echo "Building and starting the Docker container..."
docker-compose up --build

# Exit gracefully when docker-compose is terminated
exit 0 