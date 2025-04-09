import gradio as gr
import sys
from quasar_agent import QuasarChatAgent

# Attempt to initialize the agent
try:
    agent = QuasarChatAgent()
except ValueError as e:
    print(f"Error initializing agent: {e}", file=sys.stderr)
    print("Please ensure the OPENROUTER_API_KEY is set in your .env file.", file=sys.stderr)
    # You might want to display an error in the Gradio UI as well
    # or exit the application if the agent is critical.
    agent = None # Set agent to None to handle it in the UI

def respond(message, history):
    """
    Generate a response using the Quasar Alpha model via OpenRouter.
    
    Args:
        message: The user's message
        history: Chat history
        
    Returns:
        str: The model's response or an error message.
    """
    if agent is None:
        return "Error: Agent could not be initialized. Check API key configuration."
    
    # Basic history formatting (optional, adjust as needed)
    # For more complex history management, you might need to process 
    # the 'history' list and format it according to the API's requirements.
    # prompt = f"Previous conversation:\n{history}\n\nUser message:\n{message}"
    # For now, just send the latest message
    return agent.generate_response(message)

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="Quasar Alpha Chat Agent",
    description="Have a conversation with the Quasar Alpha language model.",
    theme="default",
    examples=[
        "What are the key features of large language models?",
        "Explain quantum computing to a 10-year-old",
        "Write a short poem about artificial intelligence",
        "What are some ethical considerations in AI development?",
    ],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

if __name__ == "__main__":
    if agent is None:
        print("Agent initialization failed. Gradio UI will start but may not be functional.", file=sys.stderr)
        # Optionally, prevent Gradio from launching if the agent is essential
        # sys.exit(1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)  # Configured for Hugging Face Spaces 