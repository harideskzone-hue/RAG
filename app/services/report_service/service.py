import time
from typing import Any

from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.domain.models.reasoning_context import ReasoningContext
from app.services.report_service.analytics import (
    AnalyticsEngine,
    NarrativeEngine,
    StatisticsEngine,
)
from app.services.report_service.exporter import ReportExporter


class ReportService:
    """
    Orchestrates report generation engines to produce data-driven reports.
    """
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.analytics_engine = AnalyticsEngine()
        self.statistics_engine = StatisticsEngine()
        self.narrative_engine = NarrativeEngine()
        self.exporter = ReportExporter()

    async def generate_report(self, context: ReasoningContext, format_type: str = "json") -> dict[str, Any]:
        start_time = time.time()
        self._publish_event("SERVICE_START", "report_service", context.user.user_id, start_time)
        
        try:
            bundle = context.evidence_bundle
            
            # 1. Analytics & Statistics
            analytics = self.analytics_engine.calculate(bundle)
            stats = self.statistics_engine.generate(bundle)
            
            # 2. Narrative
            narrative = self.narrative_engine.generate(bundle, analytics, stats)
            
            # 3. Export
            report_data = {
                "analytics": analytics,
                "statistics": stats,
                "narrative": narrative,
                "timeline": bundle.get_timeline()
            }
            
            exported_content = self.exporter.export(report_data, format_type)
            
            if format_type == "json":
                report_uri = None
            else:
                raise NotImplementedError("Saving reports to storage requires a configured storage client.")
            
            result = {
                "report_uri": report_uri,
                "narrative": narrative,
                "data": report_data
            }
            
            self._publish_event("SERVICE_COMPLETE", "report_service", context.user.user_id, start_time, end_time=time.time())
            return result
            
        except Exception as e:
            self._publish_event("SERVICE_ERROR", "report_service", context.user.user_id, start_time, end_time=time.time(), error=str(e))
            raise

    def _publish_event(self, event_type: str, agent_name: str, trace_id: str, start_time: float, end_time: float = None, error: str = None):
        kwargs = {
            "agent_name": agent_name,
            "event_type": event_type,
            "start_time": start_time,
            "status": "ERROR" if error else "SUCCESS",
            "trace_id": trace_id,
        }
        if end_time:
            kwargs["end_time"] = end_time
            kwargs["latency_ms"] = (end_time - start_time) * 1000
        if error:
            kwargs["errors"] = [error]
            
        self.event_bus.publish(AgentEvent(**kwargs))
