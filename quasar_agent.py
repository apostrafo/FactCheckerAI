import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import time

# Try to load environment variables from .env file, but don't fail if it doesn't exist
load_dotenv(dotenv_path=".env", verbose=False)

def fetch_available_models(max_price=0, min_context_tokens=0, cache_ttl=3600):
    """
    Fetch available models from OpenRouter API, focusing on free models.
    
    Args:
        max_price (float): Maximum price per 1M tokens (0 for free models)
        min_context_tokens (int): Minimum context length in tokens (0 = no filter)
        cache_ttl (int): Cache time-to-live in seconds
    
    Returns:
        dict: Dictionary of available models with details
    """
    # Check if we have a cached result that's still valid
    cache_file = "models_cache.json"
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                # Check if cache is still valid
                if (cache_data.get('timestamp', 0) + cache_ttl) > time.time():
                    print("Using cached model list")
                    return cache_data.get('models', {})
    except Exception as e:
        print(f"Error reading cache: {e}")
    
    try:
        # Try to fetch models directly from OpenRouter
        print("Fetching available models from OpenRouter...")
        url = "https://openrouter.ai/api/v1/models"
        print(f"Making request to: {url}")
        
        # Add a User-Agent header to avoid potential blocks
        headers = {"User-Agent": "QuasarAgent/1.0"}
        response = requests.get(url, headers=headers)
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            model_count = len(data.get('data', []))
            print(f"Received {model_count} models from OpenRouter")
            
            # Build model dictionary with actual model IDs as keys
            filtered_models = {}
            for model in data.get('data', []):
                model_id = model.get('id')
                if not model_id:
                    continue
                    
                pricing = model.get('pricing', {})
                context_length = model.get('context_length', 0)
                
                # Only include models with input pricing <= max_price and sufficient context length
                input_price = pricing.get('input', 0)
                if input_price <= max_price and context_length >= min_context_tokens:
                    # Get model size in billions of parameters (if available)
                    model_name = model.get('name', '')
                    param_size = ""
                    # Try to extract model size from name (e.g., "70B", "13B")
                    import re
                    size_match = re.search(r'(\d+(\.\d+)?)B', model_name)
                    if size_match:
                        param_size = size_match.group(0)
                    
                    # Format context length for display
                    context_display = f"{context_length/1000:.0f}K" if context_length < 1000000 else f"{context_length/1000000:.1f}M"
                    
                    # Get creation date if available
                    created_at = model.get('created_at', '')
                    date_display = ""
                    if created_at:
                        try:
                            from datetime import datetime
                            date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            date_display = date_obj.strftime("%b %Y")
                        except Exception:
                            # If date parsing fails, use original string or empty
                            if len(created_at) > 10:  # Truncate long date strings
                                date_display = created_at[:10]
                            else:
                                date_display = created_at
                    
                    # Get provider name for better organization
                    provider = "Unknown"
                    if "/" in model_id:
                        provider = model_id.split("/")[0].capitalize()
                    
                    # Create a rich display name
                    rich_name = model.get('name', model_id)
                    if param_size and param_size not in rich_name:
                        rich_name = f"{rich_name} ({param_size})"
                    
                    # Display name for UI dropdown - include context and date if available
                    display_info = []
                    if context_display:
                        display_info.append(f"Ctx: {context_display}")
                    if date_display:
                        display_info.append(f"{date_display}")
                    
                    display_name = f"{provider}: {rich_name}"
                    if display_info:
                        display_name += f" - {', '.join(display_info)}"
                                            
                    # Store model details - use the FULL model ID as the key
                    filtered_models[model_id] = {
                        'id': model_id,
                        'name': model.get('name', ''),
                        'rich_name': rich_name,
                        'display_name': display_name,
                        'context_length': context_length,
                        'param_size': param_size,
                        'created_at': date_display,
                        'description': model.get('description', ''),
                        'provider': provider
                    }
            
            # Cache the results
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'timestamp': time.time(),
                        'models': filtered_models
                    }, f)
            except Exception as e:
                print(f"Error writing cache: {e}")
                
            return filtered_models
        else:
            print(f"Error fetching models: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {}

class MultiModelAgent:
    """
    A multi-model agent that can dynamically select and query different models from OpenRouter.
    """
    
    def __init__(self, api_keys=None, max_retries=3, timeout=180):
        """
        Initialize the MultiModelAgent.
        
        Args:
            api_keys (dict): Dictionary mapping model IDs or prefixes to API keys.
                             If a key matches a prefix of a model ID, it will be used.
            max_retries (int): Maximum number of retries for failed requests
            timeout (int): Timeout in seconds for API requests
        """
        self.api_keys = api_keys or {}
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Set default OpenRouter API key from environment if not provided
        default_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_keys.get("default") and default_key:
            self.api_keys["default"] = default_key
        
        print("Initializing MultiModelAgent...")
        print(f"API Keys provided for: {list(self.api_keys.keys())}")
        
        # Fetch available models (either from cache or API)
        self.available_models = fetch_available_models(max_price=0, min_context_tokens=0)
        print(f"Loaded {len(self.available_models)} available models")
        
    def get_model_list(self):
        """
        Get the list of available model IDs.
        
        Returns:
            list: List of model IDs
        """
        return list(self.available_models.keys())
    
    def get_model_display_names(self):
        """
        Get a dictionary mapping model IDs to display names.
        
        Returns:
            dict: Mapping of model IDs to display names
        """
        return {model_id: details.get('display_name', model_id) 
                for model_id, details in self.available_models.items()}
    
    def _get_api_key_for_model(self, model_id):
        """
        Select the appropriate API key for a model.
        
        Args:
            model_id (str): The model ID
            
        Returns:
            str: The API key to use
        """
        # First check for exact match
        if model_id in self.api_keys:
            return self.api_keys[model_id]
        
        # Then check for provider prefix match (e.g., "anthropic/" for "anthropic/claude-3-opus")
        if "/" in model_id:
            provider = model_id.split("/")[0]
            if provider in self.api_keys:
                return self.api_keys[provider]
        
        # Finally, use default key
        if "default" in self.api_keys:
            return self.api_keys["default"]
            
        # If no key found, use first available key as fallback
        if self.api_keys:
            return next(iter(self.api_keys.values()))
            
        # No keys available
        raise ValueError("No API key available. Please set the OPENROUTER_API_KEY environment variable.")
    
    def generate_response(self, prompt, model=None, temperature=0.7, max_tokens=1024):
        """
        Generate a response from the specified model.
        
        Args:
            prompt (str): The prompt to send to the model
            model (str): The model ID to use
            temperature (float): Temperature for sampling (0-1)
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: The model's response text
        """
        if not model:
            raise ValueError("No model specified. Please provide a valid model ID.")
            
        # Ensure we're using a full model ID
        if model not in self.available_models:
            # Try to find the model by display name
            for model_id, details in self.available_models.items():
                if details.get('display_name') == model:
                    model = model_id
                    break
            # If still not found, raise error with diagnostic info
            if model not in self.available_models:
                # Create an error message with available models for debugging
                available_models_info = "\nAvailable models:\n"
                for idx, (m_id, details) in enumerate(list(self.available_models.items())[:10]):
                    display = details.get('display_name', m_id)
                    available_models_info += f"- {display} (ID: {m_id})\n"
                if len(self.available_models) > 10:
                    available_models_info += f"... and {len(self.available_models) - 10} more."
                    
                raise ValueError(f"Model '{model}' not found. {available_models_info}")
            
        # Get API key for this model
        try:
            api_key = self._get_api_key_for_model(model)
        except ValueError as e:
            return f"Error: {str(e)}"
            
        # Build request data
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://quasaragent.vercel.app/",
            "X-Title": "QuasarAgent",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make the request with retries
        response_text = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"Sending request to model {model} (attempt {attempt+1}/{self.max_retries})")
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                
                if response.status_code == 200:
                    response_data = response.json()
                    choices = response_data.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        response_text = message.get("content", "")
                        break
                else:
                    error_detail = "Unknown error"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get("error", {}).get("message", str(error_data))
                    except:
                        error_detail = response.text[:100] + ("..." if len(response.text) > 100 else "")
                        
                    last_error = f"HTTP {response.status_code}: {error_detail}"
                    print(f"Error from OpenRouter: {last_error}")
                    
                    # If we got a 404, the model might not exist, so don't retry
                    if response.status_code == 404:
                        break
                        
                    # Exponential backoff
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # 1, 2, 4, 8, ... seconds
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    
            except requests.Timeout:
                last_error = "Request timeout"
                print(f"Request timed out after {self.timeout} seconds")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                last_error = str(e)
                print(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
        
        if response_text:
            return response_text
        else:
            if last_error:
                return f"Error: {last_error}"
            else:
                return "Error: Failed to get a response from the model after multiple attempts."

# For backward compatibility
QuasarChatAgent = MultiModelAgent

if __name__ == "__main__":
    try:
        agent = MultiModelAgent()
        agent.chat()
    except ValueError as e:
        print(f"Initialization failed: {e}")
        print("Please create a .env file in the project directory with:")
        print("OPENROUTER_API_KEY=your_api_key_here") 