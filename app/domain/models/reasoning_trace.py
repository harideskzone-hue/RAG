from typing import Any
from pydantic import BaseModel, Field

from app.domain.models.enums import ReasoningStage

class ReasoningTrace(BaseModel):
    current_stage: ReasoningStage | None = None
    completed_stages: list[ReasoningStage] = Field(default_factory=list)
    failed_stage: ReasoningStage | None = None
    retry_count: int = 0
    logs: list[str] = Field(default_factory=list)

    def transition(self, next_stage: ReasoningStage):
        if self.current_stage and self.current_stage not in self.completed_stages:
            self.completed_stages.append(self.current_stage)
        self.current_stage = next_stage

    def fail(self, error: str):
        self.failed_stage = self.current_stage
        self.current_stage = ReasoningStage.FAILED
        self.logs.append(f"FAILED at {self.failed_stage}: {error}")
