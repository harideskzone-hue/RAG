from __future__ import annotations

import logging
from dataclasses import dataclass

from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import ModelCapabilities, LLMRequest
from app.infrastructure.llm.errors import LLMProviderError, LLMModelUnavailableError
from app.config.llm import get_model_for_role

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    provider: str
    model_id: str
    capabilities: ModelCapabilities


KNOWN_MODELS = {
    "llama-3.3-70b-versatile": ModelSpec(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=False,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=True,
            json_mode=True,
            structured_output=True,
            parallel_tools=True,
        )
    ),
    "llama-3.1-8b-instant": ModelSpec(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=False,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=True,
            json_mode=True,
            structured_output=True,
            parallel_tools=True,
        )
    ),
    "qwen-2.5-32b": ModelSpec(
        provider="groq",
        model_id="qwen-2.5-32b",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=False,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=True,
            json_mode=True,
            structured_output=True,
            parallel_tools=True,
        )
    ),
    "llama-3.2-11b-vision-preview": ModelSpec(
        provider="groq",
        model_id="llama-3.2-11b-vision-preview",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=True,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=True,
            json_mode=True,
            structured_output=True,
            parallel_tools=True,
        )
    ),
    "qwen/qwen3.6-27b": ModelSpec(
        provider="groq",
        model_id="qwen/qwen3.6-27b",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=True,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=True,
            json_mode=True,
            structured_output=True,
            parallel_tools=True,
        )
    ),
    "openai/gpt-oss-20b": ModelSpec(
        provider="groq",
        model_id="openai/gpt-oss-20b",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=False,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=True,
            json_mode=True,
            structured_output=True,
            parallel_tools=True,
        )
    ),
    "openai/gpt-oss-120b": ModelSpec(
        provider="groq",
        model_id="openai/gpt-oss-120b",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=False,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=True,
            json_mode=True,
            structured_output=True,
            parallel_tools=False,
        )
    ),
    "ollama_default": ModelSpec(
        provider="ollama",
        model_id="llama3.2:3b",
        capabilities=ModelCapabilities(
            text_input=True,
            image_input=False,
            audio_input=False,
            video_input=False,
            reasoning=True,
            tool_use=False,
            json_mode=True,
            structured_output=True,
            parallel_tools=False,
        )
    ),
}

class CapabilityError(LLMProviderError):
    pass


class CapabilityValidator:
    """Validates if a model supports the requested operations."""

    @staticmethod
    def validate(capabilities: ModelCapabilities, request: LLMRequest):
        # Determine if request contains images
        has_image = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(c, dict) and c.get("type") == "image_url" for c in msg["content"])
            for msg in request.messages
        )
        if has_image and not capabilities.image_input:
            raise CapabilityError("Model does not support image input.")

        # Determine if tools are requested
        if request.tools and not capabilities.tool_use:
            raise CapabilityError("Model does not support tool calling.")
            
        # Determine if structured output is requested
        if request.response_format and not capabilities.structured_output:
            raise CapabilityError("Model does not support structured output JSON format.")


class ModelRegistry:
    """
    Registry for LLM providers and models.
    Resolves role configuration to a specific ModelSpec, validates capabilities,
    and returns an instantiated provider.
    """

    @staticmethod
    def get_client(role: str | None = None, provider: str | None = None, model: str | None = None) -> BaseLLMClient:
        if role and not provider and not model:
            provider, model = get_model_for_role(role)
            
        if provider == "disabled" or provider == "none":
            return _DisabledLLMClient()

        # Retrieve ModelSpec
        spec = KNOWN_MODELS.get(model)
        
        # If unknown, assume a basic text model but emit warning
        if not spec:
            logger.warning(f"Unknown model '{model}'. Assuming basic text capabilities.")
            spec = ModelSpec(
                provider=provider or "groq",
                model_id=model,
                capabilities=ModelCapabilities()
            )

        client = ModelRegistry._instantiate_client(spec)
        # Wrap the client in a capability validator
        return ValidatingLLMClient(client, spec.capabilities)

    @staticmethod
    def _instantiate_client(spec: ModelSpec) -> BaseLLMClient:
        if spec.provider == "groq":
            from app.infrastructure.llm.groq_client import GroqLLMClient
            return GroqLLMClient(model=spec.model_id)
        elif spec.provider == "ollama":
            from app.infrastructure.llm.ollama_client import OllamaLLMClient
            return OllamaLLMClient(model=spec.model_id)
        elif spec.provider == "local":
            from app.infrastructure.llm.local_client import LocalLLMClient
            return LocalLLMClient(model_name=spec.model_id)
        else:
            raise LLMModelUnavailableError(f"Unknown LLM provider: '{spec.provider}'")

    @staticmethod
    def get_capabilities(model: str) -> ModelCapabilities:
        spec = KNOWN_MODELS.get(model)
        if spec:
            return spec.capabilities
        return ModelCapabilities()


class ValidatingLLMClient(BaseLLMClient):
    """Wraps an LLM client and applies capability validation before delegating."""

    def __init__(self, target_client: BaseLLMClient, capabilities: ModelCapabilities):
        self.target_client = target_client
        self._capabilities = capabilities

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(self, request: LLMRequest, **kwargs):
        CapabilityValidator.validate(self._capabilities, request)
        return await self.target_client.generate(request, **kwargs)


class _DisabledLLMClient(BaseLLMClient):
    async def generate(self, request: LLMRequest, **kwargs):
        raise LLMModelUnavailableError("LLM is explicitly disabled via configuration.")

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            text_input=False,
            reasoning=False,
            structured_output=False
        )
