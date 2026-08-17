"""
LLM Error Hierarchy.

Agents catch LLMError (the base class) — they never need to know
whether the backend is Groq, Ollama, or a local model.
"""


class LLMError(Exception):
    """Base LLM error. Agents catch this, not provider-specific exceptions."""
    pass


class LLMProviderError(LLMError):
    """Provider returned an error (rate limit, server error, etc.)."""
    pass


class LLMTimeoutError(LLMError):
    """Request timed out."""
    pass


class LLMAuthenticationError(LLMError):
    """Invalid API key or credentials."""
    pass


class LLMModelUnavailableError(LLMError):
    """Requested model is not available or not installed."""
    pass


class LLMStructuredOutputError(LLMError):
    """LLM output did not conform to the requested schema."""
    pass
