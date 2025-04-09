import gradio as gr
import sys
import asyncio
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

# Global variable to store the last responses from each model
last_responses = {"quasar": "", "deepseek": "", "gemini": ""}
# Global variable to store the last user prompt
last_prompt = ""

def get_model(model_name):
    """
    Get a model from the agent
    """
    if not agent:
        raise ValueError("Agent not initialized")
    return agent.get_model(model_name)

def generate_all_responses(message):
    """
    Generate responses from all three models simultaneously
    """
    global last_responses, last_prompt
    
    # Store the user prompt for later use in fact-checking
    last_prompt = message
    
    if not models_available:
        error_msg = "Error: Agent could not be initialized. Check API key configuration."
        return error_msg, error_msg, error_msg, gr.Button.update(interactive=False)
    
    try:
        # Get responses from all models
        responses = agent.generate_multi_response(message)
        
        # Return responses for each model (or error message if a model failed)
        quasar_response = responses.get("quasar", "Error: Failed to get response from Quasar")
        deepseek_response = responses.get("deepseek", "Error: Failed to get response from DeepSeek")
        gemini_response = responses.get("gemini", "Error: Failed to get response from Gemini")
        
        # Store the responses for later fact-checking
        last_responses = {
            "quasar": quasar_response,
            "deepseek": deepseek_response,
            "gemini": gemini_response
        }
        
        # Enable the "Check facts" button if we have valid responses
        if not any("Error" in resp for resp in last_responses.values()):
            return quasar_response, deepseek_response, gemini_response, gr.Button.update(interactive=True, variant="primary")
        else:
            return quasar_response, deepseek_response, gemini_response, gr.Button.update(interactive=False)
    
    except Exception as e:
        print(f"Error generating responses: {e}", file=sys.stderr)
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, error_msg, gr.Button.update(interactive=False)

def evaluate_response(response_text, evaluator_model_name):
    """
    Evaluate the precision of a response using another model
    """
    if not response_text:
        return "N/A"
    
    try:
        evaluator = get_model(evaluator_model_name)
        evaluation_prompt = f"""
        You are evaluating the precision of an AI's response to this user query:
        
        USER QUERY: {last_prompt}
        
        AI RESPONSE TO EVALUATE:
        {response_text}
        
        On a scale of 0-100, what percentage of the response contains factually correct information?
        Please respond with ONLY a number between 0 and 100. Do not include any other text or explanation.
        """
        
        evaluation = evaluator.generate(evaluation_prompt).strip()
        
        # Extract just the number if there's any other text
        import re
        number_match = re.search(r'\b\d{1,3}\b', evaluation)
        if number_match:
            return number_match.group(0) + "%"
        return evaluation + "%"
    except Exception as e:
        return f"Error: {str(e)}"

def check_facts():
    """
    Cross-evaluate the precision of each model's response by other models
    """
    if not last_prompt or not any(last_responses.values()):
        return "No responses to evaluate", "No responses to evaluate", "No responses to evaluate"
    
    results = {
        "quasar": {"deepseek": "", "gemini": ""},
        "deepseek": {"quasar": "", "gemini": ""},
        "gemini": {"quasar": "", "deepseek": ""}
    }
    
    # For each model's response, get evaluations from the other two models
    for model, response in last_responses.items():
        if not response:
            continue
            
        for evaluator in last_responses.keys():
            if evaluator != model:
                results[model][evaluator] = evaluate_response(response, evaluator)
    
    # Format results as HTML tables
    quasar_eval = f"""<div class="eval-container">
        <h3>How other models rate Quasar's accuracy</h3>
        <table>
        <tr><th>Model</th><th>Precision Rating</th></tr>
        <tr><td>DeepSeek</td><td>{results['quasar']['deepseek']}</td></tr>
        <tr><td>Gemini</td><td>{results['quasar']['gemini']}</td></tr>
        </table>
    </div>"""
    
    deepseek_eval = f"""<div class="eval-container">
        <h3>How other models rate DeepSeek's accuracy</h3>
        <table>
        <tr><th>Model</th><th>Precision Rating</th></tr>
        <tr><td>Quasar</td><td>{results['deepseek']['quasar']}</td></tr>
        <tr><td>Gemini</td><td>{results['deepseek']['gemini']}</td></tr>
        </table>
    </div>"""
    
    gemini_eval = f"""<div class="eval-container">
        <h3>How other models rate Gemini's accuracy</h3>
        <table>
        <tr><th>Model</th><th>Precision Rating</th></tr>
        <tr><td>Quasar</td><td>{results['gemini']['quasar']}</td></tr>
        <tr><td>DeepSeek</td><td>{results['gemini']['deepseek']}</td></tr>
        </table>
    </div>"""
    
    return quasar_eval, deepseek_eval, gemini_eval

def generate_response(prompt, model_choice):
    """
    Generate a response from the selected model
    """
    global last_prompt, last_responses
    
    last_prompt = prompt
    
    # Reset last_responses when a new prompt is submitted
    if model_choice == "All Models":
        last_responses = {"quasar": "", "deepseek": "", "gemini": ""}
    
    try:
        if model_choice == "All Models":
            return generate_all_responses(prompt)
        else:
            model = get_model(model_choice.lower())
            response = model.generate(prompt)
            # Store the response
            last_responses[model_choice.lower()] = response
            return response, "", "", gr.Button.update(interactive=True)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, error_msg, gr.Button.update(interactive=False)

# Create a Gradio interface with three output panels
with gr.Blocks(css="style.css") as app:
    gr.Markdown("# QuasarAgent - Compare AI Models")
    gr.Markdown("Generate responses from multiple AI models and check how accurately they evaluate each other's facts.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(placeholder="Enter your prompt here...", label="Prompt")
            model_choice = gr.Radio(
                ["All Models", "Quasar", "DeepSeek", "Gemini"], 
                label="Select Model", 
                value="All Models"
            )
            submit_btn = gr.Button("Generate Response")
            fact_check_btn = gr.Button("Check Facts", interactive=False)
            gr.Markdown("*Generate a response first, then click 'Check Facts' to see how models evaluate each other's factual accuracy.*", elem_id="helper-text")
    
    with gr.Row():
        with gr.Column():
            quasar_output = gr.HTML(label="Quasar AI", elem_id="quasar-output")
        with gr.Column():
            deepseek_output = gr.HTML(label="DeepSeek AI", elem_id="deepseek-output")
        with gr.Column():
            gemini_output = gr.HTML(label="Gemini AI", elem_id="gemini-output")

    with gr.Row():
        with gr.Column():
            quasar_eval = gr.HTML(label="Quasar Evaluation", visible=False)
        with gr.Column():
            deepseek_eval = gr.HTML(label="DeepSeek Evaluation", visible=False)
        with gr.Column():
            gemini_eval = gr.HTML(label="Gemini Evaluation", visible=False)
    
    submit_btn.click(
        fn=generate_response,
        inputs=[prompt, model_choice],
        outputs=[quasar_output, deepseek_output, gemini_output, fact_check_btn]
    )
    
    fact_check_btn.click(
        fn=check_facts,
        inputs=[],
        outputs=[quasar_eval, deepseek_eval, gemini_eval],
    ).then(
        fn=lambda: (gr.HTML.update(visible=True), gr.HTML.update(visible=True), gr.HTML.update(visible=True)),
        inputs=[],
        outputs=[quasar_eval, deepseek_eval, gemini_eval]
    )

if __name__ == "__main__":
    app.launch(share=True) 