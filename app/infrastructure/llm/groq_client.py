"""
Groq LLM Client.

Wraps the Groq API (or any OpenAI-compatible endpoint) behind BaseLLMClient.
All Groq-specific errors are translated into the LLM error hierarchy.

Config:
    GROQ_API_KEY          — required
    LLM_MODEL             — default: "openai/gpt-oss-120b"
    GROQ_BASE_URL         — default: "https://api.groq.com/openai/v1"
    GROQ_TIMEOUT          — default: 60 (seconds)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator

import httpx

from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMResponse, ModelCapabilities, LLMRequest
from app.infrastructure.llm.errors import (
    LLMAuthenticationError,
    LLMModelUnavailableError,
    LLMProviderError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class GroqLLMClient(BaseLLMClient):
    """
    Groq-specific implementation of the LLM interface.
    Uses the OpenAI-compatible chat completions API.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.timeout = int(os.getenv("GROQ_TIMEOUT", "60"))

    async def generate(self, request: LLMRequest, **kwargs) -> LLMResponse:
        """Invoke Groq chat completions API."""
        if not self.api_key:
            raise LLMAuthenticationError("GROQ_API_KEY is not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.reasoning_effort in ("low", "medium", "high"):
            payload["reasoning_effort"] = request.reasoning_effort
        if request.response_format:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        start = time.monotonic()
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                latency = (time.monotonic() - start) * 1000

                if resp.status_code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(1.2 * (attempt + 1))
                    continue

                if resp.status_code == 401:
                    raise LLMAuthenticationError("Invalid GROQ_API_KEY")
                if resp.status_code == 404:
                    raise LLMModelUnavailableError(f"Model '{self.model}' not found on Groq")
                if resp.status_code == 429:
                    raise LLMProviderError(f"Groq rate limit exceeded: {resp.text}")
                if resp.status_code >= 500:
                    raise LLMProviderError(f"Groq server error ({resp.status_code}): {resp.text}")
                if resp.status_code != 200:
                    raise LLMProviderError(f"Groq error ({resp.status_code}): {resp.text}")

                data = resp.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})

                return LLMResponse(
                    content=content,
                    model=data.get("model", self.model),
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    },
                    latency_ms=latency,
                )

            except (LLMAuthenticationError, LLMModelUnavailableError, LLMProviderError):
                raise
            except httpx.TimeoutException as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise LLMTimeoutError(f"Groq request timed out after {self.timeout}s") from e
                await asyncio.sleep(1.0)
            except httpx.ConnectError as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise LLMProviderError(f"Failed to connect to Groq at {self.base_url}") from e
                await asyncio.sleep(1.0)
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise LLMProviderError(f"Groq request failed: {e}") from e
                await asyncio.sleep(1.0)

        raise LLMProviderError(f"Groq request failed after {max_retries} attempts: {last_error}")

    def capabilities(self) -> ModelCapabilities:
        from app.infrastructure.llm.model_registry import ModelRegistry
        return ModelRegistry.get_capabilities(self.model)
