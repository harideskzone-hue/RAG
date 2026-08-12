from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class ConfidenceGateVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        threshold = manifest.confidence_gate.get("threshold", 0.75)
        
        if isinstance(payload, dict) and "confidence_score" in payload:
            score = payload["confidence_score"]
            if score < threshold:
                return False, f"Confidence score {score} is below threshold {threshold}."
                
        return True, "Confidence check passed."
