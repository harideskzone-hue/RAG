from uuid import uuid4
from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult
from app.domain.models.enums import EntityType, RelationshipType
from app.domain.models.relationship import Relationship


class Correlator:
    """
    Deterministic Graph Correlator.
    Cross-references entities and events across different modalities.
    """
    def run(self, context: ReasoningContext) -> EngineResult:
        entities = context.entities
        new_correlations = 0
        relationships = context.relationships

        new_relationships = []

        # 1. Identity Correlation: Link entities of the same type that share a distinct identifier
        persons = [e for e in entities if e.type == EntityType.PERSON]
        for i, p1 in enumerate(persons):
            for p2 in persons[i+1:]:
                # Check if they share an identifying attribute (e.g., person_id or name)
                if p1.attributes.get("person_id") and p1.attributes.get("person_id") == p2.attributes.get("person_id"):
                    # Ensure relationship doesn't already exist
                    exists = any(
                        (r.source_id == p1.entity_id and r.target_id == p2.entity_id) or
                        (r.source_id == p2.entity_id and r.target_id == p1.entity_id)
                        for r in relationships + new_relationships
                    )
                    if not exists:
                        # Get evidence IDs from the evidence bundle if available
                        evidence_ids = []
                        if context.evidence_bundle and context.evidence_bundle.evidence:
                            evidence_ids = [ev.evidence_id for ev in context.evidence_bundle.evidence]
                        rel = Relationship(
                            relationship_id=uuid4(),
                            source_id=p1.entity_id,
                            target_id=p2.entity_id,
                            type=RelationshipType.IDENTITY,
                            confidence=0.95,  # High confidence for deterministic identity match, but not hardcoded to 1.0
                            evidence_ids=evidence_ids
                        )
                        new_relationships.append(rel)

        # In a real implementation, we would inject these back into the graph
        if getattr(context, "relationships", None) is not None:
            context.relationships.extend(new_relationships)

        return EngineResult(
            success=True,
            partial_output={"new_correlations": len(new_relationships)}
        )