import gradio as gr
import sys
from quasar_agent import MultiModelAgent

# Attempt to initialize the agent
try:
    agent = MultiModelAgent()
    models_available = True
    model_names = list(agent.AVAILABLE_MODELS.keys())
except ValueError as e:
    print(f"Error initializing agent: {e}", file=sys.stderr)
    print("Please ensure the OPENROUTER_API_KEY is set in your .env file.", file=sys.stderr)
    agent = None
    models_available = False
    model_names = ["quasar", "deepseek", "gemini"]

def generate_all_responses(message):
    """
    Generate responses from all three models simultaneously
    """
    if not models_available:
        error_msg = "Error: Agent could not be initialized. Check API key configuration."
        return error_msg, error_msg, error_msg
    
    try:
        # Get responses from all models
        responses = agent.generate_multi_response(message)
        
        # Return responses for each model (or error message if a model failed)
        quasar_response = responses.get("quasar", "Error: Failed to get response from Quasar")
        deepseek_response = responses.get("deepseek", "Error: Failed to get response from DeepSeek")
        gemini_response = responses.get("gemini", "Error: Failed to get response from Gemini")
        
        return quasar_response, deepseek_response, gemini_response
    
    except Exception as e:
        print(f"Error generating responses: {e}", file=sys.stderr)
        return f"Error: {str(e)}", f"Error: {str(e)}", f"Error: {str(e)}"

# Create a Gradio interface with three output panels
with gr.Blocks(title="Model Comparison") as demo:
    gr.Markdown("# AI Model Comparison")
    gr.Markdown("Compare responses from Quasar Alpha, DeepSeek, and Gemini models")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Quasar Alpha")
            quasar_output = gr.Textbox(label="", lines=15)
        
        with gr.Column():
            gr.Markdown("### DeepSeek")
            deepseek_output = gr.Textbox(label="", lines=15)
            
        with gr.Column():
            gr.Markdown("### Gemini")
            gemini_output = gr.Textbox(label="", lines=15)
    
    with gr.Row():
        with gr.Column(scale=10):
            user_input = gr.Textbox(
                placeholder="Type your message here...",
                label="Your Question",
                lines=2
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Ask Models")
    
    examples = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the main differences between Python and JavaScript?", 
        "How can we address climate change effectively?",
        "Describe the ethical implications of autonomous vehicles."
    ]
    
    gr.Examples(examples=examples, inputs=user_input)
    
    # Set up event handler
    submit_btn.click(
        generate_all_responses,
        inputs=[user_input],
        outputs=[quasar_output, deepseek_output, gemini_output]
    )
    
    user_input.submit(
        generate_all_responses,
        inputs=[user_input],
        outputs=[quasar_output, deepseek_output, gemini_output]
    )

if __name__ == "__main__":
    demo.launch(share=True) 