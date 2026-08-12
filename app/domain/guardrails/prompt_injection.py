from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class PromptInjectionVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        if not manifest.prompt_injection.get("enabled", True):
            return True, "Prompt injection check disabled."
            
        text = str(payload).lower()
        if "ignore all previous instructions" in text or "system prompt" in text:
            return False, "Prompt injection pattern detected."
            
        return True, "No prompt injection detected."
