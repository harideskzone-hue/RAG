from pydantic import BaseModel
from typing import Any

class EvaluationTrace(BaseModel):
    """A unified trace that aggregates the Execution Ledger, Policy Trace, and Memory Statistics."""
    test_id: str
    execution_id: str
    duration_ms: float
    planner_trace: Any | None = None
    policy_trace: Any | None = None
    graph_statistics: Any | None = None
    memory_statistics: Any | None = None
    final_result: Any | None = None
