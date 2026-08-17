import pytest
import os
import asyncio
from unittest.mock import MagicMock

from app.cv.pipeline.video_pipeline import VideoPipeline
from app.agents.evidence_fusion.agent import EvidenceFusionAgent
from app.agents.verification.agent import VerificationAgent
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.schemas.context import VistaContext, UserContext
from app.domain.evidence import VideoEvidence, EvidenceBundle
from datetime import datetime, timezone

TEST_VIDEO_PATH = "input/VIDEO-2026-08-13-14-20-13.mp4"
MODEL_DIR = "models"
DETECTOR_MODEL = "yolo26n.pt"

def wrap_contract_to_evidence(contract) -> VideoEvidence:
    """Wraps CV EvidenceContract into RAG BaseEvidence for fusion pipeline."""
    return VideoEvidence(
        evidence_id=str(contract.evidence_id),
        source=contract.source,
        confidence=contract.confidence,
        timestamp=datetime.now(timezone.utc),
        metadata={
            "origin": {
                "type": "video_analysis",
                "video_id": contract.provenance.video_id,
                "camera_id": contract.provenance.camera_id,
                "track_id": contract.subject.track_id,
                "video_timestamp_sec": contract.provenance.video_timestamp_sec
            },
            "bbox": contract.observation.get("bbox")
        }
    )

@pytest.fixture
def pipeline_env():
    os.environ["CV_MODEL_DIR"] = MODEL_DIR
    os.environ["CV_DETECTOR_MODEL"] = DETECTOR_MODEL
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline = VideoPipeline(crop_dir=tmpdirname)
        yield pipeline

@pytest.mark.asyncio
async def test_cv_to_rag_e2e_and_mutation(pipeline_env):
    """Checklist item 13: CV -> existing RAG count E2E + Mutation test"""
    pipeline = pipeline_env
    contracts = pipeline.process_video(video_path=TEST_VIDEO_PATH, video_id="E2E_VID", camera_id="FRONT_LOBBY_CAM_01")
    
    unique_tracks_cv = len(set([c.subject.track_id for c in contracts if c.subject.track_id]))
    
    fusion_agent = EvidenceFusionAgent()
    verification_agent = VerificationAgent()
    coordinator = ResponseCoordinator()
    
    # Setup context
    query = "How many people are in the video?"
    
    class MockIntentResult:
        class QueryIntent:
            operation = "count"
            target_type = "person"
            semantic_constraints = []
            attributes = []
        query_intent = QueryIntent()

    user_ctx = UserContext(user_id="cli_user", role="admin")
    context = VistaContext(user=user_ctx)
    context.active_video_id = "E2E_VID"
    context.current_query = query
    context.results["intent_agent"] = MockIntentResult()
    
    # Wrap contracts
    base_evidences = [wrap_contract_to_evidence(c) for c in contracts]
    context.evidence_bundle = EvidenceBundle(evidence=base_evidences)
    
    # Execute RAG Loop (Deterministic portions)
    await fusion_agent.execute(context, None)
    await verification_agent.execute(context, None)
    
    # Validate CV -> RAG Count logic
    verified_contract = context.results.get("verified_contract")
    assert verified_contract is not None, "Verification failed to produce a contract"
    
    # verified_contract might be a Pydantic model or dict. Handle both.
    if isinstance(verified_contract, dict):
        verified_count = verified_contract.get("verified_count", -1)
    else:
        verified_count = getattr(verified_contract, "verified_count", -1)
    
    # The Agentic RAG verified count MUST equal the CV unique track ID count
    assert verified_count == unique_tracks_cv, f"Mismatch: CV had {unique_tracks_cv} tracks, RAG verified {verified_count}"
    print(f"CV tracks ({unique_tracks_cv}) == RAG verified count ({verified_count})")
    
    # Now run response generation with actual evidence
    response_normal = coordinator.generate_response(context)
    answer_normal = response_normal.get("final_answer", "")
    assert str(verified_count) in answer_normal, f"Normal answer didn't contain correct count: {answer_normal}"
    
    # === TEST 1: VERIFIED CONTRACT PROPAGATION TEST ===
    # Mutate contract to 999, ensuring downstream components correctly read the mutated contract
    if isinstance(verified_contract, dict):
        context.results["verified_contract"]["verified_count"] = 999
    else:
        setattr(context.results["verified_contract"], "verified_count", 999)
    
    response_propagated = coordinator.generate_response(context)
    answer_propagated = response_propagated.get("final_answer", "")
    
    assert "999" in answer_propagated, f"Propagation test failed. Expected 999 in answer, got: {answer_propagated}"
    print("Verified contract propagation test passed successfully!")
    
    # === TEST 2: LLM AUTHORITY-BOUNDARY TEST (ADVERSARIAL HALLUCINATION) ===
    # Reset contract back to actual count
    if isinstance(verified_contract, dict):
        context.results["verified_contract"]["verified_count"] = verified_count
    else:
        setattr(context.results["verified_contract"], "verified_count", verified_count)
    
    # Simulate a Reasoning LLM output that explicitly hallucinates a count, camera, and track ID
    class MockReasoningAgentResult:
        answer = "I see 999 people. Track P9999 was detected on CAM_99."
        status = "completed"
        metadata = {}
    
    context.results["reasoning_agent"] = MockReasoningAgentResult()
    
    response_adversarial = coordinator.generate_response(context)
    answer_adversarial = response_adversarial.get("final_answer", "")
    
    # The final answer must NOT adopt the hallucinated elements that contradict the contract
    assert "999" not in answer_adversarial, "Adversarial test failed! LLM hallucinated count (999) was accepted."
    assert "P9999" not in answer_adversarial, "Adversarial test failed! LLM hallucinated track (P9999) was accepted."
    assert "CAM_99" not in answer_adversarial, "Adversarial test failed! LLM hallucinated camera (CAM_99) was accepted."
    assert str(verified_count) in answer_adversarial, f"Adversarial test failed! Authoritative count ({verified_count}) missing."
    print("LLM authority-boundary (Adversarial Hallucination) test passed successfully!")

