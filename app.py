import gradio as gr
import sys
import asyncio
from quasar_agent import MultiModelAgent

# Attempt to initialize the agent
try:
    agent = MultiModelAgent(default_model="quasar_alpha")
    models_available = True
    model_names = ["quasar_alpha", "deepseek_chat_v3", "nemotron_ultra_253b"]
except ValueError as e:
    print(f"Error initializing agent: {e}", file=sys.stderr)
    print("Please ensure the OPENROUTER_API_KEY is set in your .env file.", file=sys.stderr)
    agent = None
    models_available = False
    model_names = ["quasar_alpha", "deepseek_chat_v3", "nemotron_ultra_253b"]

# Global variable to store the last responses from each model
last_responses = {"quasar_alpha": "", "deepseek_chat_v3": "", "nemotron_ultra_253b": ""}
# Global variable to store the last user prompt
last_prompt = ""

def get_model(model_name):
    """
    Get a model from the agent
    """
    if not agent:
        raise ValueError("Agent not initialized")
    # No need to get a model instance - we already have the agent that can generate responses
    return model_name

def generate_all_responses(message):
    """
    Generate responses from all three models simultaneously
    """
    global last_responses, last_prompt
    
    # Store the user prompt for later use in fact-checking
    last_prompt = message
    
    if not models_available:
        error_msg = "Error: Agent could not be initialized. Check API key configuration."
        return error_msg, error_msg, error_msg, gr.update(interactive=False)
    
    try:
        # Get responses from all models
        responses = agent.generate_multi_response(message)
        
        # Return responses for each model (or error message if a model failed)
        quasar_response = responses.get("quasar_alpha", "Error: Failed to get response from Quasar Alpha")
        deepseek_response = responses.get("deepseek_chat_v3", "Error: Failed to get response from DeepSeek Chat v3")
        nemotron_response = responses.get("nemotron_ultra_253b", "Error: Failed to get response from Nemotron Ultra 253B")
        
        # Store the responses for later fact-checking
        last_responses = {
            "quasar_alpha": quasar_response,
            "deepseek_chat_v3": deepseek_response,
            "nemotron_ultra_253b": nemotron_response
        }
        
        # Enable the "Check facts" button if we have valid responses
        if not any("Error" in resp for resp in last_responses.values()):
            return quasar_response, deepseek_response, nemotron_response, gr.update(interactive=True)
        else:
            return quasar_response, deepseek_response, nemotron_response, gr.update(interactive=False)
    
    except Exception as e:
        print(f"Error generating responses: {e}", file=sys.stderr)
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, error_msg, gr.update(interactive=False)

def evaluate_response(response_text, evaluator_model_name):
    """
    Evaluate the precision of a response using another model
    """
    if not response_text:
        return "N/A"
    
    try:
        evaluator_name = get_model(evaluator_model_name)
        evaluation_prompt = f"""
        You are evaluating the precision of an AI's response to this user query:
        
        USER QUERY: {last_prompt}
        
        AI RESPONSE TO EVALUATE:
        {response_text}
        
        On a scale of 0-100, what percentage of the response contains factually correct information?
        Please respond with ONLY a number between 0 and 100. Do not include any other text or explanation.
        """
        
        # Use the agent directly to generate the evaluation
        evaluation = agent.generate_response(evaluation_prompt, model=evaluator_name).strip()
        
        # Extract just the number if there's any other text
        import re
        number_match = re.search(r'\b\d{1,3}\b', evaluation)
        if number_match:
            return number_match.group(0) + "%"
        return evaluation + "%"
    except Exception as e:
        print(f"Error evaluating response: {e}", file=sys.stderr)
        return f"Error: {str(e)}"

def check_facts():
    """
    Cross-evaluate the precision of each model's response by other models
    """
    print("Check facts function called!")
    
    if not last_prompt or not any(last_responses.values()):
        print("No responses to evaluate")
        return "No responses to evaluate", "No responses to evaluate", "No responses to evaluate"
    
    results = {}
    
    # Print information for debugging
    print(f"Checking facts for prompt: {last_prompt}")
    print(f"Last responses: {last_responses}")
    
    try:
        results = {
            "quasar_alpha": {"deepseek_chat_v3": "", "nemotron_ultra_253b": ""},
            "deepseek_chat_v3": {"quasar_alpha": "", "nemotron_ultra_253b": ""},
            "nemotron_ultra_253b": {"quasar_alpha": "", "deepseek_chat_v3": ""}
        }
        
        # Check if any model had an error in its response
        model_errors = {}
        for model, response in last_responses.items():
            if not response:
                model_errors[model] = "No response received"
            elif response.startswith("Error"):
                model_errors[model] = response
        
        # For each model's response, get evaluations from the other two models
        for model, response in last_responses.items():
            if model in model_errors:
                print(f"Skipping {model} due to error: {model_errors[model]}")
                continue
                
            for evaluator in last_responses.keys():
                if evaluator != model:
                    if evaluator in model_errors:
                        results[model][evaluator] = f"Evaluator error: {model_errors[evaluator]}"
                        print(f"Cannot use {evaluator} to evaluate {model}: {model_errors[evaluator]}")
                    else:
                        print(f"Getting {evaluator}'s evaluation of {model}'s response")
                        results[model][evaluator] = evaluate_response(response, evaluator)
                        print(f"Result: {results[model][evaluator]}")
        
        # Format results as HTML tables
        if "quasar_alpha" in model_errors:
            quasar_eval = f"""<div class="eval-container error">
                <h3>Quasar Alpha model error</h3>
                <p>{model_errors['quasar_alpha']}</p>
            </div>"""
        else:
            quasar_eval = f"""<div class="eval-container">
                <h3>How other models rate Quasar Alpha's accuracy</h3>
                <table>
                <tr><th>Model</th><th>Precision Rating</th></tr>
                <tr><td>DeepSeek Chat v3</td><td>{results['quasar_alpha']['deepseek_chat_v3']}</td></tr>
                <tr><td>Nemotron Ultra 253B</td><td>{results['quasar_alpha']['nemotron_ultra_253b']}</td></tr>
                </table>
            </div>"""
        
        if "deepseek_chat_v3" in model_errors:
            deepseek_eval = f"""<div class="eval-container error">
                <h3>DeepSeek Chat v3 model error</h3>
                <p>{model_errors['deepseek_chat_v3']}</p>
            </div>"""
        else:
            deepseek_eval = f"""<div class="eval-container">
                <h3>How other models rate DeepSeek Chat v3's accuracy</h3>
                <table>
                <tr><th>Model</th><th>Precision Rating</th></tr>
                <tr><td>Quasar Alpha</td><td>{results['deepseek_chat_v3']['quasar_alpha']}</td></tr>
                <tr><td>Nemotron Ultra 253B</td><td>{results['deepseek_chat_v3']['nemotron_ultra_253b']}</td></tr>
                </table>
            </div>"""
        
        if "nemotron_ultra_253b" in model_errors:
            nemotron_eval = f"""<div class="eval-container error">
                <h3>Nemotron Ultra 253B model error</h3>
                <p>{model_errors['nemotron_ultra_253b']}</p>
            </div>"""
        else:
            nemotron_eval = f"""<div class="eval-container">
                <h3>How other models rate Nemotron Ultra 253B's accuracy</h3>
                <table>
                <tr><th>Model</th><th>Precision Rating</th></tr>
                <tr><td>Quasar Alpha</td><td>{results['nemotron_ultra_253b']['quasar_alpha']}</td></tr>
                <tr><td>DeepSeek Chat v3</td><td>{results['nemotron_ultra_253b']['deepseek_chat_v3']}</td></tr>
                </table>
            </div>"""
        
        print("Finished fact checking, returning results")
        return quasar_eval, deepseek_eval, nemotron_eval
    except Exception as e:
        error_msg = f"Error checking facts: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg, error_msg, error_msg

# Create a Gradio interface with a simplified layout
with gr.Blocks(css="style.css") as app:
    gr.Markdown("# QuasarAgent - Compare AI Models")
    gr.Markdown("Generate responses from multiple models and compare their fact checking abilities")
    
    with gr.Row():
        prompt = gr.Textbox(placeholder="Enter your prompt here...", label="Prompt", lines=3)
    
    with gr.Row():
        submit_btn = gr.Button("Generate Responses")
        check_facts_btn = gr.Button("Check Facts", interactive=False)
    
    with gr.Row():
        with gr.Column():
            quasar_output = gr.Textbox(label="Quasar Alpha", lines=10)
        with gr.Column():
            deepseek_output = gr.Textbox(label="DeepSeek Chat v3", lines=10)
        with gr.Column():
            nemotron_output = gr.Textbox(label="Nemotron Ultra 253B", lines=10)

    with gr.Row():
        with gr.Column():
            quasar_eval = gr.HTML(label="Quasar Alpha Evaluation")
        with gr.Column():
            deepseek_eval = gr.HTML(label="DeepSeek Chat v3 Evaluation")
        with gr.Column():
            nemotron_eval = gr.HTML(label="Nemotron Ultra 253B Evaluation")
    
    submit_btn.click(
        fn=generate_all_responses,
        inputs=[prompt],
        outputs=[quasar_output, deepseek_output, nemotron_output, check_facts_btn]
    )
    
    check_facts_btn.click(
        fn=check_facts,
        inputs=[],
        outputs=[quasar_eval, deepseek_eval, nemotron_eval]
    )

if __name__ == "__main__":
    print("Starting QuasarAgent UI...")
    app.launch(share=True) 