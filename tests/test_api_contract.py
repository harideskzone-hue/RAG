import pytest
from app.api.schemas.response import ChatResponse
from app.api.presenters.chat_presenter import ChatPresenter

@pytest.mark.asyncio
async def test_chat_response_contract_valid():
    """
    Ensures that when a RAG graph successfully completes with grounding,
    the ChatPresenter maps it correctly to the UI contract.
    """
    mock_state = {
        "status": "success",
        "final_answer": "There are 5 people.",
        "grounding_valid": True,
        "overall_confidence": 0.95,
        "evidence": [
            {
                "evidence_id": "e123",
                "source": "video_db",
                "camera_id": "CAM_01",
                "timestamp": "2026-08-15T12:00:00Z",
                "description": "Person tracked",
                "confidence": 0.99
            }
        ],
        "timeline": [{"event": "entered"}],
        "processing": {"intent": "SEARCH"},
        "execution": {
            "status": "completed",
            "steps": [{"name": "response_coord", "status": "completed"}]
        }
    }
    
    response_model = ChatPresenter.present(mock_state, execution_id="trace_123", processing_time_ms=100)
    
    assert isinstance(response_model, ChatResponse)
    assert response_model.status == "SUCCESS"
    assert response_model.grounding_status == "VALID"
    assert response_model.answer == "There are 5 people."
    assert len(response_model.evidence) == 1
    assert response_model.evidence[0].camera_id == "CAM_01"
    assert len(response_model.timeline) == 1

@pytest.mark.asyncio
async def test_chat_response_contract_abstain():
    """
    Ensures that when grounding fails or RAG abstains,
    the status is ABSTAIN and the ungrounded text doesn't mistakenly become VALID.
    """
    mock_state = {
        "status": "success", # Agent may think it succeeded
        "final_answer": "I found 999 people.", # Hallucinated
        "grounding_valid": False,
        "abstain_reason": "Grounding Validator rejected LLM response",
        "overall_confidence": 0.0,
        "evidence": [],
    }
    
    response_model = ChatPresenter.present(mock_state, execution_id="trace_456")
    
    assert response_model.status == "ABSTAIN"
    assert response_model.grounding_status == "ABSTAIN"
    
@pytest.mark.asyncio
async def test_evidence_immutability():
    """
    The evidence array MUST be strictly typed and match authoritative data.
    """
    mock_state = {
        "final_answer": "I found P1234",
        "evidence": [
            {
                "evidence_id": "e_999",
                "source": "Qdrant",
                "confidence": 0.88
            }
        ]
    }
    
    response = ChatPresenter.present(mock_state, execution_id="trace_789")
    
    # Asserting evidence structure exactly matches the Pydantic schema
    ev = response.evidence[0]
    assert ev.evidence_id == "e_999"
    assert ev.source == "Qdrant"
    assert ev.confidence == 0.88
