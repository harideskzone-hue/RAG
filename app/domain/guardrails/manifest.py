from pydantic import BaseModel

class GuardrailManifest(BaseModel):
    hallucination: dict[str, bool] = {"enabled": True}
    pii: dict[str, bool] = {"enabled": True}
    tool_permissions: dict[str, bool] = {"enabled": True}
    confidence_gate: dict[str, float] = {"threshold": 0.75}
    evidence_verification: dict[str, bool] = {"required": True}
    prompt_injection: dict[str, bool] = {"enabled": True}
