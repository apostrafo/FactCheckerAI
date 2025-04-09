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
            return quasar_response, deepseek_response, gemini_response, gr.update(interactive=True)
        else:
            return quasar_response, deepseek_response, gemini_response, gr.update(interactive=False)
    
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
            "quasar": {"deepseek": "", "gemini": ""},
            "deepseek": {"quasar": "", "gemini": ""},
            "gemini": {"quasar": "", "deepseek": ""}
        }
        
        # For each model's response, get evaluations from the other two models
        for model, response in last_responses.items():
            if not response:
                print(f"No response for {model}, skipping")
                continue
                
            for evaluator in last_responses.keys():
                if evaluator != model:
                    print(f"Getting {evaluator}'s evaluation of {model}'s response")
                    results[model][evaluator] = evaluate_response(response, evaluator)
                    print(f"Result: {results[model][evaluator]}")
        
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
        
        print("Finished fact checking, returning results")
        return quasar_eval, deepseek_eval, gemini_eval
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
            quasar_output = gr.Textbox(label="Quasar AI", lines=10)
        with gr.Column():
            deepseek_output = gr.Textbox(label="DeepSeek AI", lines=10)
        with gr.Column():
            gemini_output = gr.Textbox(label="Gemini AI", lines=10)

    with gr.Row():
        with gr.Column():
            quasar_eval = gr.HTML(label="Quasar Evaluation")
        with gr.Column():
            deepseek_eval = gr.HTML(label="DeepSeek Evaluation")
        with gr.Column():
            gemini_eval = gr.HTML(label="Gemini Evaluation")
    
    submit_btn.click(
        fn=generate_all_responses,
        inputs=[prompt],
        outputs=[quasar_output, deepseek_output, gemini_output, check_facts_btn]
    )
    
    check_facts_btn.click(
        fn=check_facts,
        inputs=[],
        outputs=[quasar_eval, deepseek_eval, gemini_eval]
    )

if __name__ == "__main__":
    print("Starting QuasarAgent UI...")
    app.launch(share=True) 