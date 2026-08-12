
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """
    Result of the Workflow Validator.
    The Supervisor consumes this to determine if execution can proceed.
    """
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    optimizations: list[str] = Field(default_factory=list)
    approval_required: bool = False
