from app.domain.event_types import EventType
from app.domain.evidence import EvidenceBundle


class RuleEngine:
    """
    Evaluates semantic event rules against the EvidenceBundle.
    """
    def evaluate(self, bundle: EvidenceBundle, intent: str) -> EventType | None:
        # A real system would use a rules engine like durable-rules or similar
        # Here we mock rule evaluation based on keywords in evidence
        
        combined_text = " ".join([
            e.metadata.get("description", "") + " " + e.metadata.get("summary", "") 
            + " " + " ".join(e.metadata.get("activities", []))
            for e in bundle.evidence
        ]).lower()
        
        if "fight" in combined_text or "punch" in combined_text:
            return EventType.FIGHT
        elif "fire" in combined_text or "smoke" in combined_text:
            return EventType.FIRE
        elif "weapon" in combined_text or "gun" in combined_text or "knife" in combined_text:
            return EventType.WEAPON
        elif "crowd" in combined_text or "gathering" in combined_text:
            return EventType.CROWD
            
        # Fallback to intent if no specific event rule triggered
        try:
            return EventType(intent)
        except ValueError:
            return EventType.UNKNOWN
