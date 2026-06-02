import os
from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # List available models
    print("📋 Available models:")
    for model in client.models.list():
        print(f"   - {model.name}")
        
except Exception as e:
    print(f"Error: {e}")