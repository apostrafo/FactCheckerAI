import os
from openai import OpenAI
from dotenv import load_dotenv

# Try to load environment variables from .env file, but don't fail if it doesn't exist
load_dotenv(dotenv_path=".env", verbose=False)

class QuasarChatAgent:
    def __init__(self, model_name="openrouter/quasar-alpha", site_url=None, site_name=None):
        """
        Initialize the Quasar Alpha chat agent via OpenRouter API.

        Args:
            model_name (str): OpenRouter model name
            site_url (str, optional): Your site URL for OpenRouter rankings.
            site_name (str, optional): Your site name for OpenRouter rankings.
        """
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        # Initialize OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Optional headers for OpenRouter ranking
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name

        print(f"QuasarChatAgent initialized for model: {self.model_name}")

    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """
        Generate a response from the model via OpenRouter API.

        Args:
            prompt (str): The input prompt
            max_length (int): Maximum response length (Note: OpenRouter might have its own limits)
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter

        Returns:
            str: The generated response or an error message.
        """
        # Note: The original example showed image input, but this implementation
        # currently only supports text input for simplicity.
        # TODO: Extend to handle multimodal input (text + images) if needed.
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
                model=self.model_name,
                messages=messages,
                max_tokens=max_length, # Note: API uses max_tokens, not max_length
                temperature=temperature,
                top_p=top_p,
                # Add other parameters as needed, e.g., stream=True for streaming
            )
            response = completion.choices[0].message.content
            return response.strip() if response else "(No response from model)"
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return f"Error: Could not get response from the model. Details: {e}"

    def chat(self):
        """
        Start an interactive chat session with the model.
        """
        print("Starting chat with Quasar Alpha (via OpenRouter). Type 'exit' to end.")
        print("=" * 50)

        while True:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            response = self.generate_response(user_input)
            print(f"Quasar Alpha: {response}")
            print("-" * 50)

if __name__ == "__main__":
    # Example: Provide optional site URL and name
    # agent = QuasarChatAgent(site_url="https://mysite.com", site_name="My Quasar App")
    try:
        agent = QuasarChatAgent()
        agent.chat()
    except ValueError as e:
        print(f"Initialization failed: {e}")
        print("Please create a .env file in the project directory with:")
        print("OPENROUTER_API_KEY=your_api_key_here") 