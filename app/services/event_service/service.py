import time
from typing import Any

from app.domain.event_types import EventType
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.domain.models.reasoning_context import ReasoningContext
from app.services.event_service.correlation import CorrelationEngine
from app.services.event_service.rule_engine import RuleEngine
from app.services.event_service.severity import SeverityEngine
from app.services.event_service.timeline import TimelineEngine


class EventService:
    """
    Orchestrates event reasoning engines to derive semantic understanding from evidence.
    """
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.rule_engine = RuleEngine()
        self.correlation_engine = CorrelationEngine()
        self.severity_engine = SeverityEngine()
        self.timeline_engine = TimelineEngine()

    async def analyze(self, context: ReasoningContext) -> dict[str, Any]:
        start_time = time.time()
        self._publish_event("SERVICE_START", "event_service", context.user.user_id, start_time)
        
        try:
            bundle = context.evidence_bundle
            intent = context.query
            
            # 1. Evaluate Rules
            event_type = self.rule_engine.evaluate(bundle, intent)
            
            # 2. Correlate Evidence
            correlations = self.correlation_engine.correlate(bundle)
            
            # 3. Determine Severity
            severity = self.severity_engine.calculate(event_type, correlations)
            
            # 4. Extract Timeline
            timeline = self.timeline_engine.extract(bundle)
            
            result = {
                "event_type": event_type.value,
                "severity": severity,
                "correlations": correlations,
                "timeline": timeline,
                "recommendations": self._generate_recommendations(event_type, severity)
            }
            
            self._publish_event("SERVICE_COMPLETE", "event_service", context.user.user_id, start_time, end_time=time.time())
            return result
            
        except Exception as e:
            self._publish_event("SERVICE_ERROR", "event_service", context.user.user_id, start_time, end_time=time.time(), error=str(e))
            raise
            
    def _generate_recommendations(self, event_type: EventType, severity: str) -> list:
        recs = []
        if severity == "CRITICAL":
            recs.append("Dispatch security immediately.")
            recs.append("Trigger lockdown protocols if applicable.")
        elif severity == "HIGH":
            recs.append("Alert nearby security personnel.")
        
        if event_type == EventType.FIRE:
            recs.append("Notify fire department.")
            
        return recs

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
