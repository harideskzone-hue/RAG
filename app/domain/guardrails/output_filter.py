from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class OutputFilterVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        text = str(payload).lower()
        if "offensive_word" in text:
            return False, "Output filter triggered."
        return True, "Output is clean."
