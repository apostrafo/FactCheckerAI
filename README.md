---
title: FactCheckerAI
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# QuasarAgent

QuasarAgent is a multi-model AI chat interface that allows users to compare responses from multiple LLM models through OpenRouter API.

## Features

- Compare responses from multiple LLM models simultaneously
- Dynamic model list pulled directly from OpenRouter API 
- Rich model information display (context length, parameter count, release date)
- Select different models for each of the 3 roles (Primary, Secondary, Evaluator)
- Automatic fact-checking with cross-model evaluation
- Responsive UI that works on mobile, tablet, and desktop
- Modern, intuitive interface with Gradio
- Docker-ready deployment

## Setup

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- OpenRouter API key

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/QuasarAgent.git
cd QuasarAgent
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenRouter API key:

```bash
echo "OPENROUTER_API_KEY=your_openrouter_api_key_here" > .env
```

## Usage

### Running locally

```bash
# On Linux/macOS
./run-local.sh

# On Windows
run-local.bat
```

### Running with Docker

```bash
# On Linux/macOS
./run-docker.sh

# On Windows
run-docker.bat
```

Then access the web interface at [http://localhost:7861](http://localhost:7861).

## Model Selection

QuasarAgent allows you to select different LLM models for 3 distinct roles:

1. **Primary Model** - The main response generator
2. **Secondary Model** - Used for comparison with the primary model's response
3. **Evaluator Model** - Used for fact-checking and evaluating the accuracy of responses

The application automatically fetches available models from OpenRouter API with their details such as context length, parameter count, and release date. The models list is dynamically updated whenever new models are added to OpenRouter. You can filter models based on your preferences and requirements.

## How It Works

1. Enter your question in the input field
2. Select models for each role using the dropdown menus (with detailed model information)
3. Click "Generate Responses" to get answers from all selected models
4. Automatic fact-checking runs to evaluate the factual accuracy of responses
5. Results are displayed in a separate tab for easy reference

## License

MIT 