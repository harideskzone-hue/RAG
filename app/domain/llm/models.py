from typing import Any, Optional
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Standardized LLM response."""
    content: str
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens
    latency_ms: float = 0.0


class ModelCapabilities(BaseModel):
    """
    Declares what a model can do.
    Used by the planner to select appropriate models for different pipeline stages.
    """
    text_input: bool = True
    image_input: bool = False
    audio_input: bool = False
    video_input: bool = False
    reasoning: bool = True
    tool_use: bool = False
    json_mode: bool = True
    structured_output: bool = True
    parallel_tools: bool = False


class LLMRequest(BaseModel):
    """Standardized request for the LLM."""
    messages: list[dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int | None = None
    reasoning_effort: str = "default"  # e.g., none, default, low, medium, high
    response_format: dict[str, Any] | None = None  # e.g. {"type": "json_object"}
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
