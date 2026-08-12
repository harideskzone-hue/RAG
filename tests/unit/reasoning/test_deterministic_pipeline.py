from uuid import uuid4
from datetime import datetime, timezone
import pytest

from app.domain.models.entity import Entity
from app.domain.models.relationship import Relationship
from app.domain.models.enums import EntityType, RelationshipType, GraphHint
from app.domain.models.reasoning_context import ReasoningContext
from app.agents.reasoning.engine.correlator import Correlator
from app.agents.reasoning.engine.contradiction_detector import ContradictionDetector
from app.agents.reasoning.engine.gap_analyzer import GapAnalyzer

def test_correlator():
    p1 = Entity(entity_id=uuid4(), type=EntityType.PERSON, attributes={"person_id": "123"})
    p2 = Entity(entity_id=uuid4(), type=EntityType.PERSON, attributes={"person_id": "123"})
    
    context = ReasoningContext(
        query="What is the connection?",
        entities=[p1, p2]
    )
    correlator = Correlator()
    result = correlator.run(context)
    
    assert result.success is True
    assert result.partial_output["new_correlations"] == 1

def test_contradiction_detector():
    p1_id = uuid4()
    
    r1 = Relationship(
        relationship_id=uuid4(),
        source_id=p1_id,
        target_id=uuid4(),
        type=RelationshipType.SEEN_AT,
        attributes={"timestamp": "2023-10-10T10:00:00Z"},
        confidence=1.0,
        evidence_ids=[]
    )
    
    r2 = Relationship(
        relationship_id=uuid4(),
        source_id=p1_id,
        target_id=uuid4(),
        type=RelationshipType.SEEN_AT,
        attributes={"timestamp": "2023-10-10T10:00:02Z"}, # Only 2 seconds apart, different cameras
        confidence=0.8,
        evidence_ids=[]
    )
    
    context = ReasoningContext(
        query="Was the car at camera 2?",
        relationships=[r1, r2]
    )
    detector = ContradictionDetector()
    result = detector.run(context)
    
    assert result.success is True
    assert len(result.partial_output["contradictions"]) == 1

def test_gap_analyzer():
    p1 = Entity(entity_id=uuid4(), type=EntityType.PERSON)
    
    r1 = Relationship(
        relationship_id=uuid4(),
        source_id=p1.entity_id,
        target_id=uuid4(),
        type=RelationshipType.SEEN_AT,
        attributes={"timestamp": 12345},
        confidence=0.9,
        evidence_ids=[]
    )
    
    context = ReasoningContext(
        query="Any missing gaps?",
        entities=[p1],
        relationships=[r1]
    )
    analyzer = GapAnalyzer()
    result = analyzer.run(context)
    
    assert result.success is True
    assert len(result.partial_output["gaps"]) == 1
    assert result.next_action is not None
    assert result.next_action.agent == "video_agent"
