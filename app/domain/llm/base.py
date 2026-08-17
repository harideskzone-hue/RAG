from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
import json
import re

from app.domain.llm.models import LLMResponse, ModelCapabilities, LLMRequest
from app.infrastructure.llm.errors import LLMStructuredOutputError


class BaseLLMClient(ABC):
    """
    Interface that all LLM providers implement.
    Agents ONLY interact through this interface.

    The `ainvoke()` method remains for backward compatibility, but
    `generate()` using `LLMRequest` is the preferred entry point.
    """

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return the capabilities of this model."""
        ...

    @abstractmethod
    async def generate(self, request: LLMRequest, **kwargs) -> LLMResponse:
        """
        Generate a response based on the LLMRequest.
        """
        ...

    async def ainvoke(self, messages: list[dict[str, Any]]) -> LLMResponse:
        """
        Backward-compatible legacy entry point.
        Wraps messages into an LLMRequest.
        """
        request = LLMRequest(messages=messages)
        return await self.generate(request)

    async def generate_structured(
        self, request: LLMRequest, schema: type, **kwargs
    ) -> Any:
        """
        Generate a structured/typed response conforming to a Pydantic schema.

        Default implementation: request JSON output, parse JSON, validate against schema.
        Providers may override with native structured output parsing.
        """
        if not request.response_format:
            request.response_format = {"type": "json_object"}

        response = await self.generate(request, **kwargs)
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response.content)
            
            # Sanitize null values that Ollama might generate for Optional/Dict fields
            def remove_none(obj):
                if isinstance(obj, dict):
                    return {k: remove_none(v) for k, v in obj.items() if v is not None}
                elif isinstance(obj, list):
                    return [remove_none(v) for v in obj if v is not None]
                return obj
            parsed = remove_none(parsed)
            
            return schema(**parsed)
        except Exception as e:
            # If Ollama hallucinates invalid nested dicts for attributes, strip them and try again
            if "attributes" in parsed:
                parsed["attributes"] = {}
            if "temporal_constraints" in parsed and not isinstance(parsed["temporal_constraints"], dict):
                parsed["temporal_constraints"] = {}
            if "spatial_constraints" in parsed and not isinstance(parsed["spatial_constraints"], dict):
                parsed["spatial_constraints"] = {}
                
            try:
                return schema(**parsed)
            except Exception as e2:
                raise LLMStructuredOutputError(
                    f"LLM output did not conform to {schema.__name__}: {e2}\n"
                    f"Raw output: {response.content[:500]}"
                ) from e2

    async def stream(self, request: LLMRequest, **kwargs) -> AsyncIterator[str]:
        """
        Stream a response token by token.
        Default implementation: yield entire response at once.
        Providers may override with true streaming support.
        """
        response = await self.generate(request, **kwargs)
        yield response.content
