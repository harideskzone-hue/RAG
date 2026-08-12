from abc import ABC, abstractmethod
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class GuardrailVerifier(ABC):
    @abstractmethod
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        """Returns (is_passed, details)"""
        pass
