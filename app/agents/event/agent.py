import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.event.schemas import EventResult
from app.domain.models.reasoning_context import ReasoningContext
from app.schemas.context import BaseResult, Citation, VistaContext
from app.domain.models import Entity, ExecutionMetadata, AgentManifest, AgentCapability, ConfidenceScore, ConfidenceFactor
from app.domain.models.enums import EntityType, AgentStatus, AgentType
from app.services.event_service.service import EventService


class EventAgent(BaseAgent):
    """
    Semantic Event Reasoning Agent.
    Orchestrates the Event Service to correlate evidence and derive events.
    """
    def __init__(self, event_service: EventService):
        self._name = "event_agent"
        self._description = "Analyzes evidence to detect and classify semantic events."
        self.service = event_service

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
                supported_intents=["EVENT_SEARCH"],
                supported_entities=[EntityType.EVENT, EntityType.INCIDENT],
                supported_modalities=["text"],
                supported_operations=["correlate", "analyze"]
            ),
            cost="medium",
            latency="fast",
            dependencies=["postgres"]
        )

    def validate(self, context: VistaContext) -> bool:
        if context.execution_plan and self.name in context.execution_plan.agents:
            return True
        return False

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> EventResult:
        start_time = time.time()

        reasoning_context = ReasoningContext(
            query=context.current_query or "Analyze event",
            user=context.user,
            evidence_bundle=context.evidence_bundle,
        )

        # Start with neutral confidence - will be updated based on actual results
        result = EventResult(
            execution_id=context.execution_id,
            trace_id=context.execution_id,
            agent_name=self.name,
            agent_type=AgentType.EVENT,
            status=AgentStatus.SUCCESS,
            confidence=ConfidenceScore(overall=0.0, factors=[]),  # Start with neutral confidence
            execution=ExecutionMetadata(duration_ms=0)
        )

        try:
            if hasattr(self.service, "analyze_events"):
                srv_res_obj = await self.service.analyze_events(reasoning_context)
                conf_score = getattr(srv_res_obj, "confidence_score", 0.8)
                result.confidence = ConfidenceScore(overall=conf_score, factors=[ConfidenceFactor(source="event_service", score=conf_score, explanation="Service event confidence")])
                result.event_type = getattr(srv_res_obj, "event_type", "UNKNOWN")
                result.severity = getattr(srv_res_obj, "severity", "LOW")
                result.correlations = getattr(srv_res_obj, "correlations", [])
                result.timeline = getattr(srv_res_obj, "timeline", [])
                result.recommendations = getattr(srv_res_obj, "recommendations", [])
            else:
                srv_res = await self.service.analyze(reasoning_context)
                conf_score = srv_res.get("confidence", 0.8)
                result.event_type = srv_res.get("event_type", "UNKNOWN")
                result.severity = srv_res.get("severity", "LOW")
                result.correlations = srv_res.get("correlations", [])
                result.timeline = srv_res.get("timeline", [])
                result.recommendations = srv_res.get("recommendations", [])
                result.confidence = ConfidenceScore(overall=conf_score, factors=[ConfidenceFactor(source="event_service", score=conf_score, explanation="Service event confidence")])

            self._last_execution_time = (time.time() - start_time) * 1000
            result.execution.duration_ms = self._last_execution_time

        except Exception as e:
            result.status = AgentStatus.ERROR
            result.metadata["error"] = str(e)

        return result

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, EventResult)

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        if isinstance(result, EventResult):
            context.agent_decisions.append({
                "agent": self.name,
                "decision": f"Detected event {result.event_type} with severity {result.severity}."
            })
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: EventResult) -> list[Citation]:
        return []

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
        }