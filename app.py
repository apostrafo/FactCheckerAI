import gradio as gr
import sys
from quasar_agent import MultiModelAgent

# Attempt to initialize the agent
try:
    agent = MultiModelAgent()
    available_models = list(agent.AVAILABLE_MODELS.keys())
except ValueError as e:
    print(f"Error initializing agent: {e}", file=sys.stderr)
    print("Please ensure the OPENROUTER_API_KEY is set in your .env file.", file=sys.stderr)
    agent = None
    available_models = []

def respond(message, history, model_choice):
    """
    Generate a response using the selected model or compare two models.
    
    Args:
        message: The user's message
        history: Chat history
        model_choice: Selected model or "compare" for comparison
        
    Returns:
        str: The model's response or error message.
    """
    if agent is None:
        return "Error: Agent could not be initialized. Check API key configuration."
    
    if model_choice == "compare":
        # Get responses from both models
        responses = agent.generate_multi_response(message)
        # Format a combined response with clearly marked sources
        result = ""
        for model_name, response in responses.items():
            result += f"## {model_name.upper()}\n\n{response}\n\n"
        return result
    else:
        # Get response from a single model
        return agent.generate_response(message, model=model_choice)

# Create a simpler Gradio interface to avoid schema issues
with gr.Blocks(title="Multi-Model Chat Agent") as demo:
    gr.Markdown("# AI Chat Agent - Compare Models")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False
                )
                submit = gr.Button("Send")
            clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            model_choices = available_models + ["compare"]
            default_model = "quasar" if "quasar" in available_models else (available_models[0] if available_models else "compare")
            model_selector = gr.Radio(
                choices=model_choices,
                value=default_model,
                label="Select Model"
            )
            
            with gr.Accordion("Available Models", open=True):
                for model in available_models:
                    gr.Markdown(f"- **{model}**: {agent.AVAILABLE_MODELS[model] if agent else 'Unknown'}")
    
    with gr.Accordion("Examples", open=False):
        gr.Examples(
            examples=[
                "What are the key features of large language models?",
                "Explain quantum computing to a 10-year-old",
                "Write a short poem about artificial intelligence",
                "What are some ethical considerations in AI development?",
            ],
            inputs=msg
        )
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history, model_choice):
        user_message = history[-1][0]
        bot_message = respond(user_message, history, model_choice)
        history[-1][1] = bot_message
        return history
    
    def clear_chat():
        return None
    
    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, model_selector], [chatbot]
    )
    clear.click(clear_chat, None, chatbot)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, model_selector], [chatbot]
    )

if __name__ == "__main__":
    if agent is None:
        print("Agent initialization failed. Gradio UI will start but may not be functional.", file=sys.stderr)
    # Updated launch parameters for Hugging Face Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True) 