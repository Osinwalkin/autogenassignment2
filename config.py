import os
from dotenv import load_dotenv

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env file or environment variables.")

LLM_CONFIG = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
            "api_key": mistral_api_key,
            "api_type": "mistral",
            "api_rate_limit": 0.1,      
            "repeat_penalty": 1.1,     
            "temperature": 0.0,        
            "seed": 42,                
            "stream": False,           
            "native_tool_calls": False,
            "cache_seed": None,        
        }
    ],
}


'''
if __name__ == "__main__":
    print("LLM_CONFIG loaded successfully:")
    print(f"  Model: {LLM_CONFIG['config_list'][0]['model']}")
    print(f"  API Type: {LLM_CONFIG['config_list'][0]['api_type']}")
    print(f"  API Key loaded: {'Yes' if LLM_CONFIG['config_list'][0]['api_key'] else 'No'}")
'''