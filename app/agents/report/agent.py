import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.report.schemas import ReportResult
from app.domain.models.reasoning_context import ReasoningContext
from app.schemas.context import BaseResult, Citation, VistaContext
from app.domain.models import Entity, ExecutionMetadata, AgentManifest, AgentCapability, ConfidenceScore, ConfidenceFactor
from app.domain.models.enums import EntityType, AgentStatus, AgentType
from app.services.report_service.service import ReportService


class ReportAgent(BaseAgent):
    """
    Data-driven Report Agent.
    Orchestrates the Report Service to generate analytics and narrative summaries.
    """
    def __init__(self, report_service: ReportService):
        self._name = "report_agent"
        self._description = "Generates comprehensive analytics reports from evidence."
        self.service = report_service

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
                supported_intents=["REPORT"],
                supported_entities=[EntityType.REPORT],
                supported_modalities=["text", "pdf", "csv"],
                supported_operations=["summarize", "export"]
            ),
            cost="low",
            latency="fast"
        )

    def validate(self, context: VistaContext) -> bool:
        if context.execution_plan and self.name in context.execution_plan.agents:
            return True
        return False

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> ReportResult:
        start_time = time.time()

        reasoning_context = ReasoningContext(
            query=context.current_query or "Generate report",
            user=context.user,
            evidence_bundle=context.evidence_bundle,
        )

        # Start with neutral confidence - will be updated based on actual results
        result = ReportResult(
            execution_id=context.execution_id,
            trace_id=context.execution_id,
            agent_name=self.name,
            agent_type=AgentType.REPORT,
            status=AgentStatus.SUCCESS,
            confidence=ConfidenceScore(overall=0.0, factors=[]),  # Start with neutral confidence
            execution=ExecutionMetadata(duration_ms=0)
        )

        try:
            format_type = "json"
            if "pdf" in reasoning_context.query.lower():
                format_type = "pdf"
            elif "csv" in reasoning_context.query.lower():
                format_type = "csv"

            srv_res = await self.service.generate_report(reasoning_context, format_type)

            if isinstance(srv_res, dict):
                result.report_uri = srv_res.get("report_uri", "")
                result.narrative = srv_res.get("narrative", "")
                result.data = srv_res.get("data", {})
                conf_score = srv_res.get("confidence", 0.9)
            else:
                result.report_uri = getattr(srv_res, "report_uri", "")
                result.narrative = getattr(srv_res, "narrative", "")
                result.data = getattr(srv_res, "data", {})
                conf_score = getattr(srv_res, "confidence", 0.9)

            result.confidence = ConfidenceScore(overall=conf_score, factors=[ConfidenceFactor(source="report_service", score=conf_score, explanation="Service report confidence")])

            self._last_execution_time = (time.time() - start_time) * 1000
            result.execution.duration_ms = self._last_execution_time

            # Map report to Entity
            if result.report_uri:
                result.entities.append(Entity(
                    type=EntityType.REPORT,
                    attributes={"report_uri": result.report_uri, "format": format_type, "narrative": result.narrative},
                    confidence=result.confidence.overall
                ))

        except Exception as e:
            result.status = AgentStatus.ERROR
            result.metadata["error"] = str(e)

        return result

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, ReportResult)

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        if isinstance(result, ReportResult):
            context.agent_decisions.append({
                "agent": self.name,
                "decision": f"Generated report at {result.report_uri}."
            })
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: ReportResult) -> list[Citation]:
        return []

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
        }