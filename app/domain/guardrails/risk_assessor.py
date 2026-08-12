from enum import Enum
from pydantic import BaseModel

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskAssessment(BaseModel):
    level: RiskLevel
    reason: str

class RiskAssessor:
    """Computes overall risk level aggregating all preceding checks."""
    @staticmethod
    def assess(failures: list[str]) -> RiskAssessment:
        if not failures:
            return RiskAssessment(level=RiskLevel.LOW, reason="No guardrail failures.")
            
        if any("jailbreak" in f.lower() or "injection" in f.lower() for f in failures):
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="Security threat detected.")
            
        if any("pii" in f.lower() or "privacy" in f.lower() for f in failures):
            return RiskAssessment(level=RiskLevel.HIGH, reason="Privacy violation detected.")
            
        return RiskAssessment(level=RiskLevel.MEDIUM, reason="Quality or confidence issues detected.")
