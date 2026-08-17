"""
Local LLM Client.

Wraps the existing LocalReasoningClient (transformers-based) behind BaseLLMClient.
Used for local model inference via HuggingFace Transformers.

Config:
    LLM_MODEL — default: "Qwen/Qwen2.5-1.5B-Instruct"
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMResponse, ModelCapabilities, LLMRequest
from app.infrastructure.llm.errors import LLMModelUnavailableError, LLMProviderError

logger = logging.getLogger(__name__)


class LocalLLMClient(BaseLLMClient):
    """
    Local model inference via HuggingFace Transformers.
    Wraps the existing LocalReasoningClient pattern.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or "Qwen/Qwen2.5-1.5B-Instruct"
        self._inner = None

    def _get_inner(self):
        """Lazy-load the actual local client."""
        if self._inner is None:
            try:
                from app.domain.llm.local_reasoning_client import LocalReasoningClient
                self._inner = LocalReasoningClient(model_name=self.model_name)
            except Exception as e:
                raise LLMModelUnavailableError(
                    f"Cannot load local model '{self.model_name}': {e}"
                ) from e
        return self._inner

    async def generate(self, request: LLMRequest, **kwargs) -> LLMResponse:
        """Invoke the local model."""
        start = time.monotonic()
        try:
            inner = self._get_inner()
            vlm_response = await inner.ainvoke(request.messages)
            latency = (time.monotonic() - start) * 1000

            return LLMResponse(
                content=vlm_response.content,
                model=self.model_name,
                usage={},
                latency_ms=latency,
            )
        except LLMModelUnavailableError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Local model inference failed: {e}") from e

    def capabilities(self) -> ModelCapabilities:
        from app.infrastructure.llm.model_registry import ModelRegistry
        return ModelRegistry.get_capabilities(self.model_name)
