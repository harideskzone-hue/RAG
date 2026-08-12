import pytest
from app.domain.guardrails.manifest import GuardrailManifest
from app.domain.guardrails.validator import GuardrailValidator
from app.domain.guardrails.prompt_injection import PromptInjectionVerifier
from app.domain.guardrails.pii import PIIVerifier
from app.domain.guardrails.risk_assessor import RiskLevel

def test_guardrail_validator_pass():
    manifest = GuardrailManifest()
    verifiers = [PromptInjectionVerifier(), PIIVerifier()]
    validator = GuardrailValidator(manifest, verifiers)
    
    payload = "This is a completely normal response."
    is_safe, risk_level, trace = validator.validate(payload)
    
    assert is_safe is True
    assert risk_level == RiskLevel.LOW
    assert trace.blocked is False
    assert len(trace.events) == 3 # 2 checks + 1 assessment

def test_guardrail_validator_fail():
    manifest = GuardrailManifest()
    verifiers = [PromptInjectionVerifier(), PIIVerifier()]
    validator = GuardrailValidator(manifest, verifiers)
    
    payload = "My SSN is 123-45-6789"
    is_safe, risk_level, trace = validator.validate(payload)
    
    assert is_safe is False
    assert risk_level == RiskLevel.HIGH
    assert trace.blocked is True
