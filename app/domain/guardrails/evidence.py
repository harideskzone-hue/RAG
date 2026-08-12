from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class EvidenceVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        if not manifest.evidence_verification.get("required", True):
            return True, "Evidence check disabled."
            
        if isinstance(payload, dict) and "citations" not in payload:
            return False, "Missing citations for factual claims."
            
        return True, "Evidence verified."
