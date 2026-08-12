from pydantic import BaseModel, Field

class ExecutionMetadata(BaseModel):
    duration_ms: float
    retries: int = 0
    model_used: str | None = None
    tool_used: list[str] = Field(default_factory=list)
    cached: bool = False
