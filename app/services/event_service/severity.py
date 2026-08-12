from app.domain.event_types import EventType


class SeverityEngine:
    """
    Determines event severity based on type and correlations.
    """
    def calculate(self, event_type: EventType, correlations: list) -> str:
        if event_type in [EventType.FIRE, EventType.WEAPON]:
            return "CRITICAL"
            
        if event_type in [EventType.FIGHT, EventType.INTRUSION]:
            return "HIGH"
            
        if event_type == EventType.CROWD:
            return "MEDIUM"
            
        return "LOW"
