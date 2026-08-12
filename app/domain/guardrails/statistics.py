from pydantic import BaseModel

class GuardrailStatistics(BaseModel):
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    total_blocked_responses: int = 0
