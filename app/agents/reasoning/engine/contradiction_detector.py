from datetime import datetime
from uuid import uuid4
from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult, Contradiction
from app.domain.models.enums import RelationshipType

class ContradictionDetector:
    """
    Deterministic Contradiction Detector.
    Uses rules and graph distances/timestamps to flag impossibilities.
    """
    def run(self, context: ReasoningContext) -> EngineResult:
        contradictions = []
        
        if context.query_engine:
            # Use Domain-specific Graph Query Engine from Phase 2
            # For each entity, find temporal conflicts
            for entity_id in context.query_engine.graph.nodes:
                conflicts = context.query_engine.find_temporal_conflicts(entity_id)
                for c1, c2 in conflicts:
                    contradictions.append(Contradiction(
                        description=f"Entity {entity_id} seen at two different locations simultaneously or impossibly fast.",
                        conflicting_evidence=[], # c1 and c2 evidence
                        severity="HIGH"
                    ))
        elif context.relationships:

            # Legacy fallback for tests
            relationships = context.relationships
        else:
            relationships = []
            
        # 1. Temporal/Spatial Impossibility
        # Example: Person seen at two distant cameras at the exact same time
        seen_at_rels = [r for r in relationships if r.type == RelationshipType.SEEN_AT]
        
        # Group by person (source_id)
        person_sightings = {}
        for rel in seen_at_rels:
            person_sightings.setdefault(rel.source_id, []).append(rel)
            
        for person_id, sightings in person_sightings.items():
            for i, s1 in enumerate(sightings):
                for s2 in sightings[i+1:]:
                    if s1.target_id != s2.target_id: # Different cameras
                        t1_str = s1.attributes.get("timestamp")
                        t2_str = s2.attributes.get("timestamp")
                        
                        if t1_str and t2_str:
                            try:
                                t1 = datetime.fromisoformat(t1_str.replace("Z", "+00:00"))
                                t2 = datetime.fromisoformat(t2_str.replace("Z", "+00:00"))
                                time_diff = abs((t2 - t1).total_seconds())
                                
                                # If seen at different cameras less than 5 seconds apart, flag contradiction
                                # (Assuming cameras are far apart for this simple rule)
                                if time_diff < 5.0:
                                    contradictions.append(Contradiction(
                                        description=f"Entity {person_id} seen at two different locations within {time_diff} seconds.",
                                        conflicting_evidence=s1.evidence_ids + s2.evidence_ids,
                                        severity="HIGH"
                                    ))
                            except ValueError:
                                pass

        return EngineResult(
            success=True,
            partial_output={"contradictions": [c.model_dump() for c in contradictions]}
        )
