import os
from openai import OpenAI
from dotenv import load_dotenv

# Try to load environment variables from .env file, but don't fail if it doesn't exist
load_dotenv(dotenv_path=".env", verbose=False)

class MultiModelAgent:
    """Agent that can interact with multiple LLM models via OpenRouter API."""
    
    # Available models
    AVAILABLE_MODELS = {
        "quasar": "openrouter/quasar-alpha",
        "deepseek": "deepseek-ai/deepseek-coder-instruct-34b"
    }
    
    def __init__(self, default_model="quasar", site_url=None, site_name=None):
        """
        Initialize the multi-model chat agent via OpenRouter API.

        Args:
            default_model (str): Default model identifier (from AVAILABLE_MODELS keys)
            site_url (str, optional): Your site URL for OpenRouter rankings.
            site_name (str, optional): Your site name for OpenRouter rankings.
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        # Initialize OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Set default model
        if default_model in self.AVAILABLE_MODELS:
            self.default_model = default_model
        else:
            self.default_model = "quasar"
            print(f"Warning: Model '{default_model}' not found, using '{self.default_model}' instead.")
        
        # Optional headers for OpenRouter ranking
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name

        print(f"MultiModelAgent initialized. Available models: {', '.join(self.AVAILABLE_MODELS.keys())}")
        print(f"Default model: {self.default_model} ({self.AVAILABLE_MODELS[self.default_model]})")

    def generate_response(self, prompt, model=None, max_length=512, temperature=0.7, top_p=0.9):
        """
        Generate a response from the specified model via OpenRouter API.

        Args:
            prompt (str): The input prompt
            model (str, optional): Model identifier (from AVAILABLE_MODELS keys). 
                                  If None, uses the default model.
            max_length (int): Maximum response length 
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter

        Returns:
            str: The generated response or an error message.
        """
        # Determine which model to use
        if model is None:
            model = self.default_model
            
        if model not in self.AVAILABLE_MODELS:
            return f"Error: Model '{model}' not found. Available models: {', '.join(self.AVAILABLE_MODELS.keys())}"
            
        model_identifier = self.AVAILABLE_MODELS[model]
            
        # Format messages for the OpenRouter API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=model_identifier,
                messages=messages,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
            )
            response = completion.choices[0].message.content
            return response.strip() if response else f"({model}: No response)"
        except Exception as e:
            print(f"Error calling OpenRouter API with model {model_identifier}: {e}")
            return f"Error from {model}: Could not get response from the model. Details: {e}"

    def generate_multi_response(self, prompt, models=None, max_length=512, temperature=0.7, top_p=0.9):
        """
        Generate responses from multiple models for the same prompt.

        Args:
            prompt (str): The input prompt
            models (list, optional): List of model identifiers to query.
                                    If None, uses all available models.
            max_length (int): Maximum response length
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter

        Returns:
            dict: Model names mapped to their responses
        """
        if models is None:
            models = list(self.AVAILABLE_MODELS.keys())
        
        results = {}
        for model in models:
            response = self.generate_response(
                prompt, 
                model=model,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            results[model] = response
            
        return results

    def chat(self):
        """
        Start an interactive command-line chat session with the default model.
        """
        print(f"Starting chat with {self.default_model} model. Type 'exit' to end the conversation.")
        print("Type '!model <name>' to switch models. Available models: " + ", ".join(self.AVAILABLE_MODELS.keys()))
        print("Type '!compare' to see responses from all models.")
        print("=" * 50)
        
        current_model = self.default_model
        compare_mode = False
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith("!model "):
                requested_model = user_input[7:].strip()
                if requested_model in self.AVAILABLE_MODELS:
                    current_model = requested_model
                    compare_mode = False
                    print(f"Switched to model: {current_model}")
                else:
                    print(f"Unknown model. Available models: {', '.join(self.AVAILABLE_MODELS.keys())}")
                continue
                
            if user_input == "!compare":
                compare_mode = not compare_mode
                print(f"Compare mode {'enabled' if compare_mode else 'disabled'}")
                continue
            
            # Generate response(s)
            if compare_mode:
                responses = self.generate_multi_response(user_input)
                print("-" * 50)
                for model_name, response in responses.items():
                    print(f"{model_name.upper()}: {response}")
                    print("-" * 50)
            else:
                response = self.generate_response(user_input, model=current_model)
                print(f"{current_model.upper()}: {response}")
                print("-" * 50)

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