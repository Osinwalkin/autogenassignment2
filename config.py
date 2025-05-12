import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env file or environment variables.")

# Define the LLM_CONFIG using the loaded API key
LLM_CONFIG = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
            "api_key": mistral_api_key,  # Use the loaded key here
            "api_type": "mistral",
            "api_rate_limit": 0.1,      # As specified in assignment
            "repeat_penalty": 1.1,     # As specified in assignment
            "temperature": 0.0,        # As specified in assignment
            "seed": 42,                # As specified in assignment
            "stream": False,           # As specified in assignment
            "native_tool_calls": False,# As specified in assignment
            "cache_seed": None,        # As specified in assignment
        }
        # You could add other model configurations here if needed
    ],
    # You might add other top-level config settings here later if needed
    # e.g., "timeout": 600,
    # "cache_seed": 42, # Global cache seed
}

# Optional: Add a check to confirm config looks okay
if __name__ == "__main__":
    print("LLM_CONFIG loaded successfully:")
    # Avoid printing the full config with the key in logs if possible
    print(f"  Model: {LLM_CONFIG['config_list'][0]['model']}")
    print(f"  API Type: {LLM_CONFIG['config_list'][0]['api_type']}")
    print(f"  API Key loaded: {'Yes' if LLM_CONFIG['config_list'][0]['api_key'] else 'No'}")