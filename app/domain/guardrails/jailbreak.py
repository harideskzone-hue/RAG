from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class JailbreakVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        text = str(payload).lower()
        if "dan " in text or "do anything now" in text:
            return False, "Jailbreak attempt detected."
        return True, "No jailbreak detected."
