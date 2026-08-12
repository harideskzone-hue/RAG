from abc import ABC, abstractmethod
from typing import Any


class BaseVLM(ABC):
    """
    Abstract VLM interface. Decouples the reasoning logic from the specific provider (e.g. Gemini, OpenAI).
    """
    @abstractmethod
    async def analyze(self, frames: list[str], prompt: str) -> dict[str, Any]:
        pass

class GeminiAdapter(BaseVLM):
    """
    Adapter for Google's Gemini Multimodal models.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name

    async def analyze(self, frames: list[str], prompt: str) -> dict[str, Any]:
        # Mocking Gemini API call
        # In reality, we'd upload frames via genai.upload_file and call model.generate_content
        return {
            "scene_summary": "A person running through the lobby.",
            "objects": ["person", "blue shirt", "backpack"],
            "activities": ["running", "entering"],
            "confidence": 0.95,
            "timeline": [{"timestamp": "0:02", "description": "Person appears"}],
            "reasoning": "The visual evidence clearly shows a person matching the description running."
        }
