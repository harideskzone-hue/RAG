import time
import uuid
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.planner.planner import ExecutionPlanner as PlanningService
from app.domain.models.confidence import ConfidenceScore

from app.domain.models import AgentManifest, AgentCapability, ExecutionMetadata
from app.domain.models.enums import AgentStatus, AgentType, ExecutionState, SchemaVersion
from app.schemas.context import BaseResult, VistaContext
from app.services.metadata_service import MetadataService
from app.domain.evidence import EvidenceBundle


class PlannerAgent(BaseAgent):
    """
    Planner Agent.
    Creates execution plans based on intent and available resources.
    """
    def __init__(self, planning_service: PlanningService, metadata_service: MetadataService):
        self._name = "planner_agent"
        self._description = "Creates execution plans for investigative workflows."
        self.planning_service = planning_service
        self.metadata_service = metadata_service

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
                supported_intents=[
                    "CAMERA_STATUS",
                    "PERSON_SEARCH",
                    "VEHICLE_SEARCH",
                    "EVENT_SEARCH",
                    "REPORT",
                    "KNOWLEDGE_GRAPH_UPDATE",
                ],
                supported_entities=[],
                supported_modalities=["text", "plan"],
                supported_operations=["plan", "optimize"],
            ),
            cost="medium",
            latency="medium",
            dependencies=["metadata_agent"],
        )

    def validate(self, context: VistaContext) -> bool:
        return (
            context.execution_plan is None
            or self.name not in context.execution_plan.agents
        )

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> BaseResult:
        start_time = time.time()
        intent = context.execution_plan.intent if context.execution_plan else ""
        entities = (
            context.results.get("intent_agent").entities
            if "intent_agent" in context.results
            else {}
        )

        # Check if we have a cached plan that's still valid
        from app.schemas.context import ExecutionPlan as _ExecutionPlan
        cached_plan = None
        if (
            hasattr(context, "execution_plan")
            and isinstance(context.execution_plan, _ExecutionPlan)
            and context.execution_plan.agents
        ):
            # If we already have a plan with our agent in it, reuse it
            if self.name in context.execution_plan.agents:
                cached_plan = context.execution_plan

        try:
            if cached_plan:
                # When plan is retrieved from cache, planning is successful
                confidence_score = ConfidenceScore(overall=0.95, factors=[])
                # We need to create a new ExecutionPlan with confidence since cached_plan might be old
                # For now, we'll update the confidence field - assuming ExecutionPlan is mutable
                cached_plan.confidence = confidence_score
                self._last_execution_time = (time.time() - start_time) * 1000
                cached_plan.execution_duration_ms = self._last_execution_time
                return BaseResult(
                    success=True,
                    evidence=[],
                    confidence=confidence_score,
                )

            intent_result = context.results.get("intent_agent")
            if not intent_result:
                from app.agents.intent.schemas import IntentResult
                from app.agents.intent.enums import Intent
                intent_result = IntentResult(
                    success=True,
                    intent=Intent(intent.lower() if intent else "unknown"),
                    confidence=ConfidenceScore(overall=1.0, factors=[]),
                    entities=entities
                )
            generated_plan = await self.planning_service.plan(
                intent_result=intent_result,
                query=getattr(context, "current_query", "") or getattr(context, "query", "")
            )

            self._last_execution_time = (time.time() - start_time) * 1000

            # Create execution metadata
            execution_metadata = ExecutionMetadata(duration_ms=self._last_execution_time)

            # Create result
            result = BaseResult(
                success=True,
                evidence=[],
                confidence=ConfidenceScore(overall=0.95, factors=[]),
            )

            # If the planning service returned a plan, attach it
            if generated_plan:
                context.execution_plan = generated_plan

            return result

        except Exception as e:
            return BaseResult(
                success=False,
                error=str(e),
                evidence=[],
                confidence=ConfidenceScore(overall=0.0, factors=[]),
            )

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, BaseResult)

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        if isinstance(result, BaseResult):
            # ResultCollector in Supervisor handles merging, but we fulfill the BaseAgent contract
            context.agent_decisions.append(
                {
                    "agent": self.name,
                    "decision": "Created execution plan.",
                }
            )
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: BaseResult) -> list:
        return []

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
            "tokens": 0,
            "tool_latency": 0.0,
            "memory_usage": 0.0,
            "errors": 0,
            "retry_count": 0,
        }