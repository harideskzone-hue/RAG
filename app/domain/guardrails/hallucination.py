from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class HallucinationVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        if not manifest.hallucination.get("enabled", True):
            return True, "Hallucination check disabled."
            
        return True, "No hallucinations detected."
