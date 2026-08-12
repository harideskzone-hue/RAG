import uuid
from datetime import datetime, timezone
import pytest

from app.domain.evidence import EvidenceBundle, MetadataEvidence
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.api.presenters.chat_presenter import ChatPresenter
from app.schemas.context import VistaContext, UserContext
from pydantic import ValidationError

def test_evidence_provenance_preservation():
    """Test 1: Full pipeline field preservation"""
    # Create bundle
    bundle = EvidenceBundle()
    ev_id = uuid.uuid4()
    ts = datetime.now(timezone.utc)
    ev = MetadataEvidence(
        evidence_id=ev_id,
        source="video_agent",
        confidence=0.42,
        timestamp=ts,
        metadata={"camera_id": "camera_01", "description": "person wearing blue shirt"}
    )
    bundle.add_evidence(ev)
    
    # Coordinator
    context = VistaContext(conversation_id="test", current_query="test", user=UserContext(user_id="test", role="admin"))
    from app.schemas.context import ExecutionPlan
    context.execution_plan = ExecutionPlan(success=True, agents=["video_agent", "reasoning_agent"])
    context.evidence_bundle = bundle
    
    class MockReasoningResult:
        def __init__(self, eids):
            self.metadata = {"claims": [{"statement": "A person was seen.", "evidence_ids": eids}], "answer": "Answer"}
            self.status = "success"
            
    context.results = {"reasoning_agent": MockReasoningResult([str(ev_id)])}
    
    coordinator = ResponseCoordinator()
    response = coordinator.generate_response(context)
    
    # Presenter
    api_response = ChatPresenter.present(response, "test-exec-id", 100)
    
    # Assertions
    assert len(api_response.evidence) == 1
    evidence_model = api_response.evidence[0]
    
    assert evidence_model.evidence_id == str(ev_id)
    assert evidence_model.source == "video_agent"
    assert evidence_model.camera_id == "camera_01"
    assert evidence_model.timestamp == ts.isoformat()
    assert evidence_model.description == "person wearing blue shirt"
    assert evidence_model.confidence == 0.42

def test_chat_presenter_isolation():
    """Test 2: ChatPresenter does not modify substantive evidence fields"""
    canonical = {
        "evidence": [
            {
                "evidence_id": "test-id",
                "source": "test-source",
                "camera_id": "test-cam",
                "timestamp": "test-time",
                "description": "test-desc",
                "confidence": 0.99
            }
        ]
    }
    api_response = ChatPresenter.present(canonical, "test")
    model = api_response.evidence[0]
    assert model.evidence_id == "test-id"
    assert model.source == "test-source"
    assert model.confidence == 0.99

def test_multiple_evidence_records_preserve_ordering():
    """Test 3: Multiple evidence records preserve ordering and IDs"""
    bundle = EvidenceBundle()
    
    ev_id1 = uuid.uuid4()
    ts1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    ev1 = MetadataEvidence(evidence_id=ev_id1, source="agent1", confidence=0.1, timestamp=ts1)
    
    ev_id2 = uuid.uuid4()
    ts2 = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)
    ev2 = MetadataEvidence(evidence_id=ev_id2, source="agent2", confidence=0.2, timestamp=ts2)
    
    # Add backwards, bundle sorts chronologically
    bundle.add_evidence(ev2)
    bundle.add_evidence(ev1)
    
    context = VistaContext(conversation_id="test", current_query="test", user=UserContext(user_id="test", role="admin"))
    from app.schemas.context import ExecutionPlan
    context.execution_plan = ExecutionPlan(success=True, agents=["agent1", "agent2", "reasoning_agent"])
    context.evidence_bundle = bundle
    
    class MockReasoningResult:
        def __init__(self, eids):
            self.metadata = {"claims": [{"statement": "Two things.", "evidence_ids": eids}], "answer": "Answer"}
            self.status = "success"
            
    context.results = {"reasoning_agent": MockReasoningResult([str(ev_id1), str(ev_id2)])}
    
    coordinator = ResponseCoordinator()
    response = coordinator.generate_response(context)
    api_response = ChatPresenter.present(response, "test-exec-id", 100)
    
    assert len(api_response.evidence) == 2
    # Should be sorted chronologically by EvidenceBundle
    assert api_response.evidence[0].evidence_id == str(ev_id1)
    assert api_response.evidence[1].evidence_id == str(ev_id2)

def test_duplicate_evidence_id_deduplication():
    """Test 4: Verify duplicate evidence IDs are handled by deduplication policy"""
    bundle = EvidenceBundle()
    ev_id = uuid.uuid4()
    ts = datetime.now(timezone.utc)
    ev1 = MetadataEvidence(evidence_id=ev_id, source="agent1", confidence=0.1, timestamp=ts)
    ev2 = MetadataEvidence(evidence_id=ev_id, source="agent2", confidence=0.2, timestamp=ts)
    
    bundle.add_evidence(ev1)
    bundle.add_evidence(ev2) # Should be skipped due to deduplication
    
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].source == "agent1"

def test_missing_provenance_validation_failure():
    """Test 5: Verify missing provenance is NOT replaced with fabricated values -> FAIL"""
    canonical = {
        "evidence": [
            {
                # Missing evidence_id
                "source": "test-source",
                "camera_id": "test-cam",
                "timestamp": "test-time",
                "description": "test-desc",
                "confidence": 0.99
            }
        ]
    }
    
    with pytest.raises(KeyError):
        # Presenter uses strict dict lookup e["evidence_id"] now
        ChatPresenter.present(canonical, "test")

def test_cross_request_evidence_isolation():
    """Test 6: Verify cross-request evidence cannot appear in current response"""
    context = VistaContext(conversation_id="req1", current_query="test", user=UserContext(user_id="test", role="admin"))
    context.evidence_bundle = EvidenceBundle()
    context.evidence_bundle.add_evidence(
        MetadataEvidence(evidence_id=uuid.uuid4(), source="src", confidence=1.0, timestamp=datetime.now(timezone.utc))
    )
    
    context2 = VistaContext(conversation_id="req2", current_query="test", user=UserContext(user_id="test", role="admin"))
    # Different context has empty bundle unless explicitly shared
    assert getattr(context2, "evidence_bundle", None) is None
    
    coordinator = ResponseCoordinator()
    response = coordinator.generate_response(context2)
    api_response = ChatPresenter.present(response, "test")
    assert len(api_response.evidence) == 0

def test_missing_reasoning_agent_blocks():
    """Test 7: Ensure that when reasoning is required but absent, the response blocks"""
    context = VistaContext(conversation_id="test", current_query="test", user=UserContext(user_id="test", role="admin"))
    from app.schemas.context import ExecutionPlan
    context.execution_plan = ExecutionPlan(success=True, agents=["reasoning_agent"])
    context.results = {}
    
    coordinator = ResponseCoordinator()
    response = coordinator.generate_response(context)
    
    assert response["status"] == "error"
    assert "Response blocked" in response["final_answer"]
    assert len(response["evidence"]) == 0

def test_reasoning_claim_without_evidence_blocks():
    """Test 8: Ensure claims that lack supporting evidence IDs result in empty evidence trace"""
    bundle = EvidenceBundle()
    ev_id = uuid.uuid4()
    bundle.add_evidence(MetadataEvidence(evidence_id=ev_id, source="src", confidence=1.0, timestamp=datetime.now(timezone.utc)))
    
    context = VistaContext(conversation_id="test", current_query="test", user=UserContext(user_id="test", role="admin"))
    context.evidence_bundle = bundle
    
    class MockReasoningResult:
        def __init__(self):
            self.metadata = {"claims": [{"statement": "Claim without evidence", "evidence_ids": []}], "answer": "Answer"}
            self.status = "success"
            
    context.results = {"reasoning_agent": MockReasoningResult()}
    
    coordinator = ResponseCoordinator()
    response = coordinator.generate_response(context)
    
    # Should succeed but have empty evidence trace
    assert len(response["evidence"]) == 0

def test_reasoning_cannot_cite_uncanonical_evidence_id():
    """Test 9: Ensure a claim citing a fake or aliased ID is blocked (results in empty evidence trace)"""
    bundle = EvidenceBundle()
    ev_id = uuid.uuid4()
    bundle.add_evidence(MetadataEvidence(evidence_id=ev_id, source="src", confidence=1.0, timestamp=datetime.now(timezone.utc)))
    
    context = VistaContext(conversation_id="test", current_query="test", user=UserContext(user_id="test", role="admin"))
    context.evidence_bundle = bundle
    
    class MockReasoningResult:
        def __init__(self):
            # Citing a hallucinated ID "E1" instead of the real UUID
            self.metadata = {"claims": [{"statement": "Hallucinated citation", "evidence_ids": ["E1", str(uuid.uuid4())]}], "answer": "Answer"}
            self.status = "success"
            
    context.results = {"reasoning_agent": MockReasoningResult()}
    
    coordinator = ResponseCoordinator()
    response = coordinator.generate_response(context)
    
    # The non-matching IDs are ignored, so the evidence trace should be empty
    assert len(response["evidence"]) == 0
