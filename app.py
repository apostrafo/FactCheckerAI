import gradio as gr
import sys
from quasar_agent import MultiModelAgent

# Attempt to initialize the agent
try:
    agent = MultiModelAgent()
    models_available = True
except ValueError as e:
    print(f"Error initializing agent: {e}", file=sys.stderr)
    print("Please ensure the OPENROUTER_API_KEY is set in your .env file.", file=sys.stderr)
    agent = None
    models_available = False

# Current model selection - global state
current_model = "quasar"  # default model

def respond(message, history):
    """
    Simple callback for the Gradio ChatInterface
    """
    if not models_available:
        return "Error: Agent could not be initialized. Check API key configuration."
    
    global current_model
    
    # Check for model switching commands
    if message.startswith("/model "):
        model_name = message[7:].strip().lower()
        if model_name in agent.AVAILABLE_MODELS or model_name == "compare":
            current_model = model_name
            return f"Switched to model: {current_model}"
        else:
            return f"Unknown model. Available models: {', '.join(agent.AVAILABLE_MODELS.keys())}, compare"
    
    # Generate response based on current model
    try:
        if current_model == "compare":
            # Get responses from both models
            responses = agent.generate_multi_response(message)
            # Format a combined response
            result = "Comparison:\n\n"
            for model_name, response in responses.items():
                result += f"## {model_name.upper()}\n{response}\n\n"
            return result
        else:
            return agent.generate_response(message, model=current_model)
    except Exception as e:
        return f"Error: {str(e)}"

# Create a minimal Gradio Interface
demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Type a message or /model [name] to switch models"),
    title="AI Chat Agent - Quasar & DeepSeek",
    description="Available commands: /model quasar, /model deepseek, /model compare",
    theme="soft"
)

if __name__ == "__main__":
    print(f"Starting with model: {current_model}")
    demo.launch(share=True) 