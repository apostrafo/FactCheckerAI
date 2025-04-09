# Quasar Alpha Chat Agent (OpenRouter API)

A chat interface for interacting with the `openrouter/quasar-alpha` language model via the OpenRouter API.

## Features

- Command-line chat interface
- Web-based GUI via Gradio
- Uses the OpenRouter API for inference
- Configurable response parameters
- Requires an OpenRouter API Key

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/QuasarAgent.git
   cd QuasarAgent
   ```
   **(Note: Replace the URL above with your actual repository URL if applicable)**

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory (`QuasarAgent/`) and add your OpenRouter API key:
   ```dotenv
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```
   Replace `your_openrouter_api_key_here` with your actual key.

## Usage

### Command-line Interface

Run the interactive chat in the terminal:

```
python quasar_agent.py
```

### Web Interface

Start the Gradio web interface:

```
python app.py
```

This will launch a local web server, and the interface will be available at http://localhost:7860. A public link will also be provided if `share=True` is enabled.

## Configuration

You can customize the model's behavior by modifying the parameters in the `generate_response` method (`quasar_agent.py`):

- `max_tokens`: Maximum number of tokens in the generated response (Note: API uses `max_tokens`)
- `temperature`: Controls randomness (higher = more random)
- `top_p`: Controls diversity via nucleus sampling

You can also provide optional `site_url` and `site_name` parameters when initializing `QuasarChatAgent` in `quasar_agent.py` or `app.py` for OpenRouter ranking purposes.

## Requirements

- Python 3.8+
- An active OpenRouter API Key

## License

This project is open-source and is provided under the MIT License. 