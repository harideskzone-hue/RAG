import pytest
import asyncio
from uuid import uuid4
from datetime import datetime, timezone

from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.entity import Entity
from app.domain.models.relationship import Relationship
from app.domain.models.enums import EntityType, RelationshipType, EvidenceType
from app.domain.evidence import EvidenceBundle, PersonEvidence
from app.agents.reasoning.service import ReasoningService
from app.agents.reasoning.agent import ReasoningAgent

@pytest.mark.asyncio
async def test_reasoning_pipeline_integration():
    # Setup Mock Graph
    p1 = Entity(entity_id=uuid4(), type=EntityType.PERSON, attributes={"person_id": "999"})
    p2 = Entity(entity_id=uuid4(), type=EntityType.PERSON, attributes={"person_id": "999"})
    
    r1 = Relationship(
        relationship_id=uuid4(),
        source_id=p1.entity_id,
        target_id=uuid4(),
        type=RelationshipType.SEEN_AT,
        attributes={"timestamp": "2023-10-10T10:00:00Z"},
        confidence=1.0,
        evidence_ids=[]
    )
    
    r2 = Relationship(
        relationship_id=uuid4(),
        source_id=p2.entity_id,
        target_id=uuid4(),
        type=RelationshipType.SEEN_AT,
        attributes={"timestamp": "2023-10-10T10:00:01Z"},
        confidence=1.0,
        evidence_ids=[]
    )
    
    # Mock State
    class MockContext:
        def __init__(self):
            self.query = "Where did the person go?"
            self.execution_id = uuid4()
            self.results = {}
            self.evidence_bundle = EvidenceBundle(evidence=[
                PersonEvidence(
                    evidence_id=uuid4(),
                    evidence_type=EvidenceType.VECTOR,
                    source="vector",
                    confidence=0.9,
                    created_at=datetime.utcnow().replace(tzinfo=timezone.utc),
                    timestamp=datetime.utcnow().replace(tzinfo=timezone.utc),
                    trace_id=uuid4(),
                    metadata={"description": "mock person"}
                )
            ])
            
    mock_context = MockContext()
    
    # Run Agent
    # Provide no LLM client to trigger the deterministic fallback logic
    service = ReasoningService(llm_client=None)
    agent = ReasoningAgent(service=service)
    
    agent_result = await agent.execute(mock_context)
    mock_context = agent.finish(mock_context, agent_result)
    
    # Assertions
    assert "reasoning" in mock_context.results
    reasoning_result = mock_context.results["reasoning"]
    
    assert reasoning_result.status == "success"
    assert "completed_stages" in reasoning_result.metadata
    completed = reasoning_result.metadata["completed_stages"]
    
    assert "correlation" in completed
    assert "contradiction" in completed
    assert "gap_analysis" in completed
    assert "hypothesis_generation" in completed
    assert "verification" in completed
