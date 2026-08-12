from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult, InformationGap, AgentExecutionRequest
from app.domain.models.enums import EntityType, RelationshipType

class GapAnalyzer:
    """
    Deterministic Gap Analyzer.
    Detects missing edges and recommends next actions to the Supervisor.
    """
    def run(self, context: ReasoningContext) -> EngineResult:
        relationships = context.relationships
        entities = context.entities
        gaps = []
        next_action = None
        
        # 1. Missing Exit Tracking
        # If a person is seen at a camera but has no subsequent tracking or exit event
        persons = [e for e in entities if e.type == EntityType.PERSON]
        
        for person in persons:
            # Get all relationships for this person
            person_rels = [r for r in relationships if r.source_id == person.entity_id]
            
            has_seen = any(r.type == RelationshipType.SEEN_AT for r in person_rels)
            has_exit = any(r.type == RelationshipType.EXITED for r in person_rels)
            has_enter = any(r.type == RelationshipType.ENTERED for r in person_rels)
            
            if has_seen and not (has_exit or has_enter):
                gap = InformationGap(
                    description=f"Person {person.entity_id} was seen but has no exit or entry record.",
                    missing_entities=["tracking_event", "exit_camera"]
                )
                
                gaps.append(gap)
        
        # Determine if we should request a new agent via the Supervisor
        next_action = None
        if gaps:
            if "video_agent" not in context.blackboard.requested_agents:
                next_action = AgentExecutionRequest(
                    agent="video_agent",
                    priority="HIGH",
                    reason="Need to track person exit or entry from last known camera.",
                    expected_output=["tracking_event", "exit_camera"]
                )
                context.blackboard.add_requested_agent("video_agent")

        return EngineResult(
            success=True,
            partial_output={"gaps": [g.model_dump() for g in gaps]},
            next_action=next_action
        )
