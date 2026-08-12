from app.domain.guardrails.verifier import GuardrailVerifier
from app.domain.guardrails.manifest import GuardrailManifest
from typing import Any

class ToolGuardVerifier(GuardrailVerifier):
    def verify(self, payload: Any, manifest: GuardrailManifest) -> tuple[bool, str]:
        if not manifest.tool_permissions.get("enabled", True):
            return True, "Tool permissions check disabled."
            
        if isinstance(payload, dict) and "requested_tool" in payload:
            if payload["requested_tool"] == "delete_database":
                return False, "Unauthorized tool access: delete_database"
                
        return True, "Tool access authorized."
