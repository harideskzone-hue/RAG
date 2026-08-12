import os
import json
import httpx
import logging
from typing import List, Dict, Any, Optional

from app.domain.llm.local_reasoning_client import VLMResponse

logger = logging.getLogger(__name__)

class OllamaReasoningClient:
    """
    Reasoning client for lightweight text inference through Ollama HTTP API.
    Expects structured JSON output matching the ReasoningResult schema.
    """
    def __init__(self):
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model = os.environ.get("OLLAMA_REASONING_MODEL", "qwen2.5:0.5b")
        self.timeout = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

    async def ainvoke(self, messages: List[Dict[str, Any]]) -> VLMResponse:
        """
        Invokes the Ollama model asynchronously and attempts to return structured JSON.
        """
        # Check if Ollama is available
        try:
            async with httpx.AsyncClient() as client:
                # We do a quick ping first to check availability
                resp = await client.get(f"{self.host}/api/tags", timeout=5.0)
                if resp.status_code != 200:
                    raise Exception(f"Ollama server returned {resp.status_code}")
                
                # Check if model exists
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                if self.model not in models:
                    logger.error(f"REASONING_MODEL_BLOCKED: Ollama model '{self.model}' not installed. Installed: {models}")
                    return VLMResponse(content=f'{{"error": "REASONING_MODEL_BLOCKED: Model {self.model} not installed"}}')
        except Exception as e:
            logger.error(f"REASONING_MODEL_BLOCKED: Ollama unavailable at {self.host} ({e})")
            return VLMResponse(content='{"error": "REASONING_MODEL_BLOCKED: Ollama unavailable"}')

        # Convert standard message format to Ollama format
        ollama_messages = []
        for m in messages:
            if isinstance(m, dict) and "role" in m and "content" in m:
                # If content is a list (multimodal), we extract text. We assume text reasoning only.
                content = m["content"]
                if isinstance(content, list):
                    text_parts = [p["text"] for p in content if p.get("type") == "text"]
                    content = "\n".join(text_parts)
                ollama_messages.append({"role": m["role"], "content": content})
            elif hasattr(m, "role") and hasattr(m, "content"):
                ollama_messages.append({"role": m.role, "content": m.content})
            else:
                ollama_messages.append({"role": "user", "content": str(m)})

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "format": "json"  # Enforce JSON generation
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.host}/api/chat", json=payload)
                
                if response.status_code != 200:
                    logger.error(f"Ollama inference failed with status {response.status_code}: {response.text}")
                    return VLMResponse(content='{"error": "REASONING_MODEL_BLOCKED: Inference failed"}')
                
                result_json = response.json()
                content = result_json.get("message", {}).get("content", "{}")
                return VLMResponse(content=content)
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return VLMResponse(content='{"error": "REASONING_MODEL_BLOCKED: Inference request failed"}')
