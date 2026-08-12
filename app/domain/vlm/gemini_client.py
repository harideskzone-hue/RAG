import os
from typing import List, Dict, Any
from app.domain.vlm.client import VLMClient, VLMResponse
from google import genai
from google.genai import types

class GeminiVLClient(VLMClient):
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model_name = model_name
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        self.client = genai.Client(api_key=api_key)

    async def ainvoke(self, messages: List[Dict[str, Any]]) -> VLMResponse:
        system_instruction = ""
        user_prompt = ""
        for m in messages:
            if m["role"] == "system":
                system_instruction += m["content"] + "\n"
            elif m["role"] == "user":
                user_prompt += m["content"] + "\n"
                
        import asyncio
        
        def _call_gemini():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )
            
        response = await asyncio.to_thread(_call_gemini)
        
        clean_text = response.text.strip()
        if clean_text.startswith('```json'):
            clean_text = clean_text[7:]
        if clean_text.startswith('```'):
            clean_text = clean_text[3:]
        if clean_text.endswith('```'):
            clean_text = clean_text[:-3]
            
        return VLMResponse(content=clean_text.strip())
