import time
from typing import Any
from uuid import uuid4

from app.agents.base_agent import BaseAgent
from app.schemas.context import BaseResult, Citation, VistaContext
from app.domain.models import AgentManifest, AgentCapability, ConfidenceScore, ConfidenceFactor, ExecutionMetadata
from app.domain.models.enums import AgentStatus, AgentType, EntityType
from app.tools.general.time_tool import TimeTool


class TimeResult(BaseResult):
    answer: str = ""
    formatted_time: str = ""
    formatted_date: str = ""

class TimeAgent(BaseAgent):
    """
    General Capability Agent for System Time & Status Queries.
    Executes TimeTool and returns direct time answers without invoking heavy CCTV RAG.
    """
    def __init__(self, time_tool: TimeTool = None):
        self._name = "time_agent"
        self._description = "Provides current wall-clock date, time, and system clock capabilities."
        self.tool = time_tool or TimeTool()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def manifest(self) -> AgentManifest:
        return AgentManifest(
            name=self.name,
            description=self.description,
            capabilities=AgentCapability(
                supported_intents=["time_query", "system_status"],
                supported_entities=[EntityType.SCENE],
                supported_modalities=["text"],
                supported_operations=["query_time"]
            ),
            cost="none",
            latency="fast",
            dependencies=[]
        )

    def validate(self, context: VistaContext) -> bool:
        if context.execution_plan and self.name in context.execution_plan.agents:
            return True
        intent_val = getattr(context.results.get("intent_agent"), "intent", None)
        if intent_val and str(getattr(intent_val, "value", intent_val)).lower() in ["time_query", "time"]:
            return True
        return False

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> TimeResult:
        start_time = time.time()
        
        tool_res = await self.tool.execute(context)
        answer = tool_res.get("answer", "Current time unavailable.")

        duration_ms = (time.time() - start_time) * 1000

        result = TimeResult(
            success=True,
            answer=answer,
            formatted_time=tool_res.get("formatted_time", ""),
            formatted_date=tool_res.get("formatted_date", ""),
            confidence=ConfidenceScore(overall=1.0, factors=[ConfidenceFactor(source="system_clock", score=1.0, explanation="System clock verified")]),
            error=None
        )

        return result

    def verify(self, result: BaseResult) -> bool:
        return result.success

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        ans = getattr(result, "answer", "")
        context.agent_decisions.append({
            "agent": self.name,
            "decision": f"Retrieved system clock: {ans}"
        })
        return context

    def confidence(self, result: BaseResult) -> float:
        return 1.0

    def citations(self, result: BaseResult) -> list[Citation]:
        return [Citation(source_type="system_clock", source_id="local_clock", content="Local System Wall-Clock", relevance_score=1.0)]

    def metrics(self) -> dict[str, Any]:
        return {"execution_time_ms": getattr(self, "_last_execution_time", 0.0)}
