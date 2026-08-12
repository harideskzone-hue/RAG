import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

try:
    for m in client.models.list_models():
        print(f"name: {m.name}")
except Exception as e:
    print(f"Error: {e}")
