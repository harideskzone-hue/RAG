from app.domain.guardrails.manifest import GuardrailManifest
from app.domain.guardrails.trace import GuardrailTrace
from app.domain.guardrails.statistics import GuardrailStatistics
from app.domain.guardrails.risk_assessor import RiskAssessor, RiskLevel
from app.domain.guardrails.verifier import GuardrailVerifier
from typing import Any
from uuid import uuid4

class GuardrailValidator:
    """Orchestrator interface that ensures nothing reaches the user without passing the enabled checks."""
    def __init__(self, manifest: GuardrailManifest, verifiers: list[GuardrailVerifier]):
        self.manifest = manifest
        self.verifiers = verifiers
        self.statistics = GuardrailStatistics()
        
    def validate(self, payload: Any) -> tuple[bool, RiskLevel, GuardrailTrace]:
        trace = GuardrailTrace(execution_id=str(uuid4()))
        failures = []
        
        for verifier in self.verifiers:
            self.statistics.total_checks += 1
            passed, details = verifier.verify(payload, self.manifest)
            
            trace.add_event(
                name=verifier.__class__.__name__,
                status="PASS" if passed else "FAIL",
                details=details
            )
            
            if not passed:
                self.statistics.failed_checks += 1
                failures.append(details)
            else:
                self.statistics.passed_checks += 1
                
        # Risk Assessment
        assessment = RiskAssessor.assess(failures)
        trace.add_event(
            name="RiskAssessor",
            status="COMPLETED",
            details=f"Assessed Risk Level: {assessment.level.value} - {assessment.reason}"
        )
        
        # Block anything HIGH or CRITICAL
        is_safe = assessment.level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        trace.blocked = not is_safe
        
        if trace.blocked:
            self.statistics.total_blocked_responses += 1
            
        return is_safe, assessment.level, trace
