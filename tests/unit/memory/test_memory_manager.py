from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.domain.evidence import EvidenceBundle, MetadataEvidence
from app.memory.manager import MemoryManager
from app.memory.policy import MemoryPolicy
from app.schemas.context import UserContext, VistaContext


@pytest.fixture
def base_context():
    return VistaContext(
        conversation_id="test_conv",
        user=UserContext(user_id="user1", role="admin"),
        messages=[{"role": "user", "content": "hello " * 10} for _ in range(50)], # 50 messages, triggers summarization
        evidence_bundle=EvidenceBundle()
    )

@pytest.mark.asyncio
async def test_conversation_summarization(base_context):
    policy = MemoryPolicy(max_messages=40)
    manager = MemoryManager(policy)
    
    # Pre-condition
    assert len(base_context.messages) == 50
    
    await manager.run(base_context)
    
    # Post-condition: Should have kept 5 + 1 summary = 6
    assert len(base_context.messages) == 6
    assert base_context.messages[0]["role"] == "system"
    assert "Summary" in base_context.messages[0]["content"]

@pytest.mark.asyncio
async def test_evidence_eviction(base_context):
    policy = MemoryPolicy(evidence_ttl_hours=12)
    manager = MemoryManager(policy)
    
    now = datetime.now(timezone.utc)
    t1 = uuid4()
    t2 = uuid4()
    old_ev_id = uuid4()
    new_ev_id = uuid4()
    crit_ev_id = uuid4()
    
    # Add old evidence (24 hours old) - Should be evicted
    base_context.evidence_bundle.add_evidence(MetadataEvidence(
        evidence_id=old_ev_id,
        source="postgres",
        confidence=1.0,
        timestamp=now - timedelta(hours=24),
        trace_id=t1
    ))
    
    # Add recent evidence (1 hour old) - Should be kept
    base_context.evidence_bundle.add_evidence(MetadataEvidence(
        evidence_id=new_ev_id,
        source="postgres",
        confidence=1.0,
        timestamp=now - timedelta(hours=1),
        trace_id=t1
    ))
    
    # Add critical old evidence (24 hours old) - Should be kept due to priority
    base_context.evidence_bundle.add_evidence(MetadataEvidence(
        evidence_id=crit_ev_id,
        source="postgres",
        confidence=1.0,
        timestamp=now - timedelta(hours=24),
        trace_id=t2,
        metadata={"priority": "critical", "camera_id": "cam_critical"}
    ))
    
    assert len(base_context.evidence_bundle.evidence) == 3
    
    await manager.run(base_context)
    
    assert len(base_context.evidence_bundle.evidence) == 2
    retained_ids = [e.evidence_id for e in base_context.evidence_bundle.evidence]
    assert new_ev_id in retained_ids
    assert crit_ev_id in retained_ids
    assert old_ev_id not in retained_ids
