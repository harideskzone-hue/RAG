from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any
import re

class PIIVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        if not manifest.pii.get("enabled", True):
            return True, "PII check disabled."
            
        text = str(payload)
        # Mock simple regex for SSN
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
            return False, "PII detected in payload."
            
        return True, "No PII detected."
