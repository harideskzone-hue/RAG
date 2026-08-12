import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.confidence.engine import ConfidenceEngine
from app.domain.models.confidence import ConfidenceResult, ConfidenceScore
from app.domain.models import AgentManifest, AgentCapability
from app.schemas.context import BaseResult, Citation, VistaContext


class ConfidenceAgent(BaseAgent):
    """
    Evaluates confidence of the EvidenceBundle using the ConfidenceEngine.
    This acts as a gatekeeper before Video reasoning or final response.
    """
    def __init__(self, engine: ConfidenceEngine):
        self._name = "confidence_agent"
        self._description = "Calculates multi-factor confidence and recommends next actions."
        self.engine = engine

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def manifest(self) -> AgentManifest:
        return AgentManifest(
            self.name,
            self.description,
            AgentCapability(supported_operations=["confidence_scoring"]),
            cost="low", latency="fast"
        )

    def validate(self, context: VistaContext) -> bool:
        return context.evidence_bundle is not None

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> ConfidenceResult:
        start_time = time.time()

        intent = context.execution_plan.intent if context.execution_plan else "UNKNOWN"
        result = self.engine.evaluate(context.evidence_bundle, intent)

        # Set the confidence field to match the overall confidence from the report
        result.confidence = ConfidenceScore(overall=result.report.overall, factors=[])

        self._last_execution_time = (time.time() - start_time) * 1000
        return result

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, ConfidenceResult)

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        # Safely cast to ConfidenceResult since this agent only produces ConfidenceResult
        if isinstance(result, ConfidenceResult):
            context.confidence_report = result.report
            context.confidence_score = result.report.overall

            context.agent_decisions.append({
                "agent": self.name,
                "decision": f"Confidence {result.report.overall:.2f}. Next action: {result.next_action}"
            })

        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: BaseResult) -> list[Citation]:
        return []

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
            "errors": 0,
            "retry_count": 0
        }