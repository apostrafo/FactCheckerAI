import gradio as gr
import sys
import asyncio
import hashlib
import json
import os
import time
import markdown
from datetime import datetime, timedelta

# Add more detailed debug prints
print("Starting app.py script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Add memory info for debugging
try:
    import psutil
    print(f"Available memory: {psutil.virtual_memory().available / (1024 * 1024):.2f} MB")
except ImportError:
    print("psutil not available for memory debugging")

try:
    print("Attempting to import MultiModelAgent...")
    from quasar_agent import MultiModelAgent
    print("MultiModelAgent import successful")
except Exception as e:
    print(f"Error importing MultiModelAgent: {e}", file=sys.stderr)
    sys.exit(1)

# Set up multiple API keys for different models
model_api_keys = {
    "default": os.getenv("OPENROUTER_API_KEY"),  # Default key from .env
    "anthropic": "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",
    "nvidia": "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"
}

print(f"Using API keys for providers:")
for provider, key in model_api_keys.items():
    masked_key = key[:10] + "..." + key[-5:] if key else "None"
    print(f"  - {provider}: {masked_key}")

# Attempt to initialize the agent
print("Attempting to initialize the agent...")
try:
    agent = MultiModelAgent(
        api_keys=model_api_keys,
        max_retries=3,
        timeout=180
    )
    print("Agent initialized successfully")
    models_available = True
    # Get available models from the agent
    available_models = agent.get_model_list()
    # Get model display names for the UI
    model_display_names = agent.get_model_display_names()
    
    # Log available models for debugging
    print("Available models:")
    for model_id in available_models:
        display_name = model_display_names.get(model_id, model_id)
        print(f"  - ID: {model_id} | Display: {display_name}")
except ValueError as e:
    print(f"Error initializing agent: {e}", file=sys.stderr)
    print("Please ensure the OPENROUTER_API_KEY is set in your .env file.", file=sys.stderr)
    agent = None
    models_available = False
    available_models = []
    model_display_names = {}
except Exception as e:
    print(f"Unexpected error initializing agent: {type(e).__name__}: {e}", file=sys.stderr)
    agent = None
    models_available = False
    available_models = []
    model_display_names = {}

# Global variable to store the last responses from each role
last_responses = {}
# Global variable to store the last user prompt
last_prompt = ""

# Add a simple cache to avoid redundant API calls
response_cache = {}
cache_ttl = timedelta(hours=24)  # Cache entries expire after 24 hours
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(cache_dir, exist_ok=True)

def get_cache_key(prompt, model_name):
    """Generate a unique cache key for a prompt + model combination"""
    key_string = f"{prompt}:{model_name}"
    return hashlib.md5(key_string.encode()).hexdigest()

def get_from_cache(prompt, model_name):
    """Try to get a response from the cache"""
    cache_key = get_cache_key(prompt, model_name)
    
    # First check memory cache
    if cache_key in response_cache:
        entry = response_cache[cache_key]
        if entry['timestamp'] + cache_ttl > datetime.now():
            print(f"Cache hit for {model_name} (memory)")
            return entry['response']
        else:
            # Expired entry
            del response_cache[cache_key]
    
    # Then check disk cache
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                entry = json.load(f)
                timestamp = datetime.fromisoformat(entry['timestamp'])
                if timestamp + cache_ttl > datetime.now():
                    # Add to memory cache
                    response_cache[cache_key] = {
                        'response': entry['response'],
                        'timestamp': timestamp
                    }
                    print(f"Cache hit for {model_name} (disk)")
                    return entry['response']
                else:
                    # Expired entry
                    os.remove(cache_file)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Cache read error: {e}")
    
    return None

def save_to_cache(prompt, model_name, response):
    """Save a response to the cache"""
    if not response or response.startswith("Error"):
        return  # Don't cache errors
        
    cache_key = get_cache_key(prompt, model_name)
    timestamp = datetime.now()
    
    # Save to memory cache
    response_cache[cache_key] = {
        'response': response,
        'timestamp': timestamp
    }
    
    # Save to disk cache
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'response': response,
                'timestamp': timestamp.isoformat()
            }, f)
    except Exception as e:
        print(f"Cache write error: {e}")

def get_model(model_name):
    """
    Get a model from the agent
    """
    if not agent:
        raise ValueError("Agent not initialized")
    # No need to get a model instance - we already have the agent that can generate responses
    return model_name

def create_model_dropdown_choices():
    """
    Create a list of (model_id, model_display_name) tuples for dropdowns
    where the model_id is both the value and the key
    """
    choices = []
    for model_id in available_models:
        display_name = model_display_names.get(model_id, model_id)
        # The key and value are both the model_id, but we show the display_name
        choices.append((model_id, display_name))
    return choices

async def generate_response_async(message, model_name):
    """
    Generate a response from a model asynchronously with caching and timing
    """
    print(f"Generating response with model: '{model_name}'")
    start_time = time.time()
    
    # Try to get response from cache first
    cached_response = get_from_cache(message, model_name)
    if cached_response:
        # For cached responses, we'll use a very small time to indicate it was from cache
        elapsed_time = 0.01  # 10ms to indicate cached response
        return cached_response, elapsed_time
        
    try:
        response = agent.generate_response(message, model=model_name)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Save successful response to cache
        if response and not response.startswith("Error"):
            save_to_cache(message, model_name, response)
            
        return response, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error generating response from {model_name}: {e}", file=sys.stderr)
        return f"Error: Failed to get response from {model_name}. Details: {str(e)}", elapsed_time

async def generate_all_responses_async(message, primary_model, secondary_model, evaluator_model):
    """
    Generate responses from all three models sequentially and run fact checking if all succeed
    """
    global last_responses, last_prompt, model_display_names
    
    # Store the user prompt for later use in fact-checking
    last_prompt = message
    
    if not models_available:
        error_msg = "Error: Agent could not be initialized. Check API key configuration."
        return error_msg, error_msg, error_msg, None
    
    try:
        # Process each model sequentially to reduce memory usage
        response_times = {}
        
        # Process primary model
        primary_name = primary_model
        print(f"Generating response from {primary_name} (primary role)...")
        primary_response, primary_time = await generate_response_async(message, primary_name)
        last_responses[primary_name] = primary_response
        response_times["primary"] = primary_time
        print(f"Primary model response generated in {primary_time:.2f}s")
        
        # Process secondary model
        secondary_name = secondary_model
        print(f"Generating response from {secondary_name} (secondary role)...")
        secondary_response, secondary_time = await generate_response_async(message, secondary_name)
        last_responses[secondary_name] = secondary_response
        response_times["secondary"] = secondary_time
        print(f"Secondary model response generated in {secondary_time:.2f}s")
        
        # Process evaluator model
        evaluator_name = evaluator_model
        print(f"Generating response from {evaluator_name} (evaluator role)...")
        evaluator_response, evaluator_time = await generate_response_async(message, evaluator_name)
        last_responses[evaluator_name] = evaluator_response
        response_times["evaluator"] = evaluator_time
        print(f"Evaluator model response generated in {evaluator_time:.2f}s")
        
        # Format responses with timing information
        primary_output = f"Model: {model_display_names.get(primary_name, primary_name)}\n\n{last_responses[primary_name]}\n\n⏱️ Response time: {response_times['primary']:.2f} seconds"
        secondary_output = f"Model: {model_display_names.get(secondary_name, secondary_name)}\n\n{last_responses[secondary_name]}\n\n⏱️ Response time: {response_times['secondary']:.2f} seconds"
        evaluator_output = f"Model: {model_display_names.get(evaluator_name, evaluator_name)}\n\n{last_responses[evaluator_name]}\n\n⏱️ Response time: {response_times['evaluator']:.2f} seconds"
        
        # Check if any responses have errors
        has_errors = any("Error" in resp for resp in last_responses.values())
        
        # If no errors, automatically run fact checking
        fact_check_results = None
        if not has_errors:
            print("All models generated responses successfully, running automatic fact checking...")
            fact_check_results = await check_facts_async()
        
        return primary_output, secondary_output, evaluator_output, fact_check_results
    
    except Exception as e:
        print(f"Error generating responses: {e}", file=sys.stderr)
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, error_msg, None

def generate_all_responses(message, primary_model, secondary_model, evaluator_model):
    """
    Synchronous wrapper for generate_all_responses_async
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(generate_all_responses_async(message, primary_model, secondary_model, evaluator_model))

async def evaluate_response_async(response_text, evaluator_model_name):
    """
    Evaluate the precision of a response using another model (async version)
    """
    start_time = time.time()
    
    if not response_text:
        return "N/A", 0
    
    try:
        # Get the evaluator model name
        evaluator_name = evaluator_model_name
        
        # Check if we already have a cached evaluation
        cache_key = get_cache_key(f"EVAL:{last_prompt}:{response_text}", evaluator_name)
        if cache_key in response_cache:
            entry = response_cache[cache_key]
            if entry['timestamp'] + cache_ttl > datetime.now():
                print(f"Evaluation cache hit for {evaluator_name}")
                return entry['response'], 0.01  # Return minimal time for cached responses
            else:
                del response_cache[cache_key]
        
        # Create a more efficient evaluation prompt with fewer tokens
        evaluation_prompt = f"""Rate factual accuracy of this AI response to: '{last_prompt}'
Response: {response_text}
Your rating (0-100) with no explanation:"""
        
        # Use the agent directly to generate the evaluation
        evaluation = agent.generate_response(evaluation_prompt, model=evaluator_name).strip()
        
        # Extract just the number if there's any other text
        import re
        number_match = re.search(r'\b\d{1,3}\b', evaluation)
        result = number_match.group(0) + "%" if number_match else evaluation + "%"
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Cache the evaluation result
        response_cache[cache_key] = {
            'response': result,
            'timestamp': datetime.now()
        }
        
        return result, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error evaluating response: {e}", file=sys.stderr)
        return f"Error: {str(e)}", elapsed_time

def evaluate_response(response_text, evaluator_model_name):
    """
    Synchronous wrapper for evaluate_response_async
    """
    return asyncio.run(evaluate_response_async(response_text, evaluator_model_name))

async def check_facts_async():
    """
    Cross-evaluate the precision of each model's response by other models (sequential version)
    Processing sequentially to reduce memory usage
    """
    print("Check facts function called!")
    
    if not last_prompt or not any(last_responses.values()):
        print("No responses to evaluate")
        return "No responses to evaluate"
    
    # Print information for debugging
    print(f"Checking facts for prompt: {last_prompt}")
    print(f"Last responses: {last_responses}")
    
    try:
        # Create dynamic structure for storing results based on current roles
        results = {}
        eval_times = {}
        
        for role in last_responses:
            results[role] = {}
            eval_times[role] = {}
            for evaluator_role in last_responses:
                if evaluator_role != role:
                    results[role][evaluator_role] = ""
                    eval_times[role][evaluator_role] = 0
        
        # Check if any model had an error in its response
        role_errors = {}
        for role, response in last_responses.items():
            if not response:
                role_errors[role] = "No response received"
            elif response.startswith("Error"):
                role_errors[role] = response
        
        # Process evaluations sequentially to reduce memory usage
        for role, response in last_responses.items():
            if role in role_errors:
                print(f"Skipping {role} due to error: {role_errors[role]}")
                continue
                
            # Strip timing information from the response (if any)
            response_text = response.split("\n\n⏱️ Response time:")[0] if "\n\n⏱️ Response time:" in response else response
            
            for evaluator_role in last_responses.keys():
                if evaluator_role != role:
                    if evaluator_role in role_errors:
                        results[role][evaluator_role] = f"Evaluator error: {role_errors[evaluator_role]}"
                        print(f"Cannot use {evaluator_role} to evaluate {role}: {role_errors[evaluator_role]}")
                    else:
                        evaluator_model = evaluator_role
                        print(f"Getting {evaluator_role} ({evaluator_model})'s evaluation of {role}'s response...")
                        result, elapsed_time = await evaluate_response_async(response_text, evaluator_model)
                        results[role][evaluator_role] = result
                        eval_times[role][evaluator_role] = elapsed_time
                        print(f"Result for {evaluator_role} evaluating {role}: {result} (took {elapsed_time:.2f}s)")
        
        # Build HTML for result display
        html_sections = []
        
        for role in last_responses:
            model_name = role
            
            if role in role_errors:
                html_sections.append(f"""<div class="eval-container error">
                    <h3>{role.capitalize()} model ({model_name}) error</h3>
                    <p>{role_errors[role]}</p>
                </div>""")
            else:
                # Build table rows for each evaluator
                table_rows = ""
                for evaluator_role in last_responses:
                    if evaluator_role != role:
                        evaluator_model = evaluator_role
                        table_rows += f"""<tr>
                            <td>{evaluator_role.capitalize()} ({evaluator_model})</td>
                            <td>{results[role][evaluator_role]}</td>
                            <td>{eval_times[role][evaluator_role]:.2f}s</td>
                        </tr>"""
                
                html_sections.append(f"""<div class="eval-container">
                    <h3>How other models rate {role.capitalize()}'s ({model_name}) accuracy</h3>
                    <table>
                    <tr><th>Model</th><th>Precision Rating</th><th>Evaluation Time</th></tr>
                    {table_rows}
                    </table>
                </div>""")
        
        # Join all sections
        full_report = "".join(html_sections)
        
        print("Finished fact checking, returning results")
        return full_report
    except Exception as e:
        error_msg = f"Error checking facts: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg

def check_facts():
    """
    Synchronous wrapper for check_facts_async
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(check_facts_async())

def launch_interface():
    print("Launching Gradio interface...")
    
    print("Preparing to create Gradio Blocks...")
    
    # Limit the number of models for performance
    limited_models = available_models[:50] if len(available_models) > 50 else available_models
    print(f"Limited to {len(limited_models)} models for performance")
    
    # Create model choices with a limited set
    def create_limited_model_choices():
        choices = []
        for model_id in limited_models:
            display_name = model_display_names.get(model_id, model_id)
            choices.append((model_id, display_name))
        return choices
    
    model_choices = create_limited_model_choices()
    print(f"Created dropdown choices with {len(model_choices)} options")
    
    try:
        with gr.Blocks(css="style.css", theme=gr.themes.Soft()) as demo:
            print("Creating Gradio UI components...")
            gr.Markdown("# AI Model Comparison")
            
            if not models_available:
                gr.Markdown("⚠️ **WARNING:** Models could not be initialized. Check API key configuration.")
            
            with gr.Row():
                # Column for inputs and controls
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        placeholder="Enter your question here...", 
                        label="Question",
                        lines=4,
                        container=True
                    )
                    
                    gr.Markdown("### Select Models to Compare")
                    
                    model1 = gr.Dropdown(
                        choices=model_choices,
                        value=None,
                        label="Model 1",
                        interactive=models_available,
                        allow_custom_value=True
                    )
                    
                    model2 = gr.Dropdown(
                        choices=model_choices,
                        value=None,
                        label="Model 2",
                        interactive=models_available,
                        allow_custom_value=True
                    )
                    
                    model3 = gr.Dropdown(
                        choices=model_choices,
                        value=None,
                        label="Model 3",
                        interactive=models_available,
                        allow_custom_value=True
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Generate & Compare", variant="primary", scale=3)
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            # Results container
            with gr.Column():
                results_html = gr.HTML(
                    "<div class='placeholder'>Results will appear here</div>"
                )
            
            print("Setting up Gradio event handlers...")
            # Set up event handlers
            submit_btn.click(
                fn=generate_and_evaluate,
                inputs=[prompt, model1, model2, model3],
                outputs=[results_html],
                show_progress=True
            )
            
            clear_btn.click(
                fn=lambda: "<div class='placeholder'>Results cleared</div>",
                inputs=None,
                outputs=[results_html]
            )
            
            print("UI setup complete, launching Gradio interface...")
        
        # Launch the interface
        print("About to call demo.launch()...")
        demo.launch(server_name="0.0.0.0", server_port=7861)
        print("Gradio launch completed.")
    except Exception as e:
        print(f"ERROR launching Gradio interface: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("Attempting simplified interface as fallback...")
        
        # Create a simple interface as fallback
        with gr.Interface(
            fn=lambda txt: f"Simplified interface due to error: {txt}",
            inputs="text",
            outputs="text",
            title="QuasarAgent Fallback Interface"
        ) as simple_demo:
            simple_demo.launch(server_name="0.0.0.0", server_port=7861)

async def generate_and_evaluate_responses(prompt, selected_models):
    """
    Generate responses from all selected models and have them evaluate each other
    
    Args:
        prompt (str): The user's question/prompt
        selected_models (list): List of selected model identifiers
        
    Returns:
        list: List of formatted responses with evaluations
    """
    global last_prompt
    last_prompt = prompt
    
    # Filter out duplicates and None values
    valid_models = []
    for model in selected_models:
        if model and model not in valid_models:
            valid_models.append(model)
    
    if not valid_models:
        return ["No valid models selected"]
    
    if not models_available or not agent:
        return ["Error: Models not available. Check API key configuration."]
    
    try:
        # Step 1: Generate responses from all models
        print("Generating responses from selected models...")
        responses = {}
        response_times = {}
        
        for model in valid_models:
            try:
                start_time = time.time()
                
                # Check cache first
                cached_response = get_from_cache(prompt, model)
                if cached_response:
                    responses[model] = cached_response
                    response_times[model] = 0.01  # Very small time to indicate cache hit
                    print(f"Cache hit for {model}")
                else:
                    # Generate new response
                    response = agent.generate_response(prompt, model=model)
                    responses[model] = response
                    elapsed_time = time.time() - start_time
                    response_times[model] = elapsed_time
                    print(f"Generated response from {model} in {elapsed_time:.2f}s")
                    
                    # Cache successful response
                    if response and not response.startswith("Error"):
                        save_to_cache(prompt, model, response)
            except Exception as e:
                print(f"Error generating response from {model}: {e}")
                responses[model] = f"Error: {str(e)}"
                response_times[model] = 0.0
        
        # Step 2: Have each model evaluate other models' responses
        print("Starting cross-evaluation of model responses...")
        evaluations = {}
        
        for evaluator in valid_models:
            # Skip models that failed to respond - they can't evaluate others
            if not responses[evaluator] or responses[evaluator].startswith("Error"):
                print(f"Skipping evaluations by {evaluator} due to response error")
                continue
                
            evaluations[evaluator] = {}
            
            for evaluated in valid_models:
                # Don't evaluate your own response
                if evaluator == evaluated:
                    continue
                
                # Skip evaluating models that failed to respond
                if not responses[evaluated] or responses[evaluated].startswith("Error"):
                    evaluations[evaluator][evaluated] = "N/A"
                    continue
                
                # Create evaluation prompt
                evaluation_prompt = f"""Rate factual accuracy of this AI response to: '{prompt}'
Response: {responses[evaluated]}
Your rating (0-100) with no explanation:"""
                
                try:
                    print(f"Getting {evaluator}'s evaluation of {evaluated}'s response...")
                    start_time = time.time()
                    
                    # Check evaluation cache first
                    cache_key = get_cache_key(f"EVAL:{prompt}:{responses[evaluated]}", evaluator)
                    cached_eval = None
                    if cache_key in response_cache:
                        entry = response_cache[cache_key]
                        if entry['timestamp'] + cache_ttl > datetime.now():
                            cached_eval = entry['response']
                    
                    if cached_eval:
                        print(f"Evaluation cache hit for {evaluator} evaluating {evaluated}")
                        rating = cached_eval
                    else:
                        # Generate new evaluation
                        evaluation = agent.generate_response(evaluation_prompt, model=evaluator)
                        
                        # Extract numeric rating with regex
                        import re
                        number_match = re.search(r'\b\d{1,3}\b', evaluation)
                        rating = number_match.group(0) if number_match else "N/A"
                        
                        # Cache the evaluation result
                        response_cache[cache_key] = {
                            'response': rating,
                            'timestamp': datetime.now()
                        }
                    
                    evaluations[evaluator][evaluated] = rating
                    print(f"{evaluator}'s rating of {evaluated}: {rating}%")
                    
                except Exception as e:
                    print(f"Error getting evaluation from {evaluator}: {e}")
                    evaluations[evaluator][evaluated] = f"N/A"
        
        # Step 3: Format results with integrated evaluations
        formatted_results = []
        
        for model in valid_models:
            # Get model display name
            display_name = model_display_names.get(model, model)
            
            # Calculate average accuracy if available
            ratings = []
            rating_details = []
            
            for other_model in valid_models:
                if other_model == model:
                    continue
                
                if other_model in evaluations and model in evaluations[other_model]:
                    rating = evaluations[other_model][model]
                    if rating not in ("N/A", "") and not rating.startswith("N/A"):
                        try:
                            ratings.append(int(rating))
                        except ValueError:
                            pass
                    rating_details.append(f"{rating}%")
                else:
                    rating_details.append("N/A")
            
            # Calculate average accuracy
            if ratings:
                avg_accuracy = sum(ratings) / len(ratings)
                accuracy_display = f"📊 Accuracy: {avg_accuracy:.0f}% ({', '.join(rating_details)})"
            else:
                accuracy_display = f"📊 Accuracy: N/A ({', '.join(rating_details)})"
            
            # Format model response with accuracy BEFORE actual response
            model_header = f"## Model: {display_name}"
            timing_info = f"⏱️ Response time: {response_times[model]:.2f} seconds"
            
            # Check if this model had an error
            if not responses[model] or responses[model].startswith("Error"):
                response_text = f"⚠️ {responses[model]}"
            else:
                response_text = responses[model]
            
            # Construct final formatted response with accuracy BEFORE response
            formatted_response = f"{model_header}\n\n{timing_info}\n{accuracy_display}\n\n{response_text}"
            formatted_results.append(formatted_response)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in generate_and_evaluate_responses: {e}")
        return [f"Error processing request: {str(e)}"]

def generate_and_evaluate(prompt, model1, model2, model3):
    """
    Synchronous wrapper for generate_and_evaluate_responses
    """
    # Filter out None values and extract model IDs (in case we got tuples from dropdowns)
    selected_models = []
    for model in [model1, model2, model3]:
        if model:
            # Extract model ID if we got a tuple (this can happen with gr.Dropdown)
            if isinstance(model, tuple) and len(model) > 0:
                model = model[0]  # First element is the model ID
            selected_models.append(model)
    
    print(f"Selected models for comparison: {selected_models}")
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    responses = loop.run_until_complete(generate_and_evaluate_responses(prompt, selected_models))
    
    # Format results as HTML
    import markdown
    
    html = "<div class='results-container'>"
    for response in responses:
        response_html = markdown.markdown(response)
        html += f"<div class='model-result'>{response_html}</div>"
    html += "</div>"
    
    return html

if __name__ == "__main__":
    print("Starting QuasarAgent app...")
    launch_interface() 