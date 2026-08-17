"""
Ollama LLM Client.

Wraps the Ollama HTTP API behind BaseLLMClient.
All Ollama-specific errors are translated into the LLM error hierarchy.

Config:
    OLLAMA_HOST              — default: "http://127.0.0.1:11434"
    OLLAMA_REASONING_MODEL   — default: "qwen2.5:0.5b"
    OLLAMA_TIMEOUT           — default: 60 (seconds)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMResponse, ModelCapabilities, LLMRequest
from app.infrastructure.llm.errors import (
    LLMModelUnavailableError,
    LLMProviderError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OllamaLLMClient(BaseLLMClient):
    """
    Ollama-specific implementation of the LLM interface.
    Uses the Ollama HTTP chat API.
    """

    def __init__(self, model: str | None = None):
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model = model or os.environ.get("OLLAMA_REASONING_MODEL", "qwen2.5:0.5b")
        self.timeout = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

    async def generate(self, request: LLMRequest, **kwargs) -> LLMResponse:
        """Invoke the Ollama model via HTTP API."""
        # Check availability and model existence
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.host}/api/tags", timeout=5.0)
                if resp.status_code != 200:
                    raise LLMProviderError(f"Ollama server returned {resp.status_code}")

                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                if self.model not in models:
                    raise LLMModelUnavailableError(
                        f"Ollama model '{self.model}' not installed. Available: {models}"
                    )
        except (LLMModelUnavailableError, LLMProviderError):
            raise
        except Exception as e:
            raise LLMProviderError(f"Ollama unavailable at {self.host}: {e}") from e

        # Convert messages to Ollama format
        ollama_messages = []
        for m in request.messages:
            if isinstance(m, dict) and "role" in m and "content" in m:
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
            "format": "json",
        }

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.host}/api/chat", json=payload)

            latency = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                raise LLMProviderError(
                    f"Ollama inference failed ({response.status_code}): {response.text}"
                )

            result_json = response.json()
            content = result_json.get("message", {}).get("content", "{}")

            return LLMResponse(
                content=content,
                model=self.model,
                usage={},  # Ollama doesn't provide token counts in the same way
                latency_ms=latency,
            )

        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Ollama request timed out after {self.timeout}s") from e
        except (LLMProviderError, LLMTimeoutError):
            raise
        except Exception as e:
            raise LLMProviderError(f"Ollama request failed: {e}") from e

    def capabilities(self) -> ModelCapabilities:
        from app.infrastructure.llm.model_registry import ModelRegistry
        return ModelRegistry.get_capabilities(self.model)
