import pytest
from unittest.mock import AsyncMock

from app.schemas.intent import QueryIntent, IntentType, EntityType
from app.graph.nodes.intent import IntentNode
from app.graph.nodes.planner import PlannerNode
from app.graph.nodes.retrieval import RetrievalNode
from app.graph.nodes.verification import VerificationNode
from app.graph.nodes.response import ResponseNode
from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMRequest, LLMResponse
from app.services.db_services import EvidenceService, TrackService

class MockE2ELLMClient(BaseLLMClient):
    def __init__(self, mock_intent, mock_response_text):
        self.mock_intent = mock_intent
        self.mock_response_text = mock_response_text
        
    def capabilities(self): pass
    
    async def generate(self, request, **kwargs):
        return LLMResponse(content=self.mock_response_text)
        
    async def generate_structured(self, request, schema, **kwargs):
        return self.mock_intent

@pytest.mark.asyncio
async def test_full_rag_pipeline_success():
    # 1. Setup mock LLM & Services
    mock_intent = QueryIntent(intent_type=IntentType.SEARCH, entity_type=EntityType.PERSON, is_valid=True, confidence=0.9)
    llm = MockE2ELLMClient(mock_intent, "The person was seen 5 times.")
    
    mock_evidence = AsyncMock(spec=EvidenceService)
    mock_evidence.search_evidence.return_value = [{"evidence_id": "e1"}, {"evidence_id": "e2"}] # Returns some evidence
    mock_track = AsyncMock(spec=TrackService)
    
    # 2. Setup Nodes
    intent_node = IntentNode(llm)
    planner_node = PlannerNode()
    retrieval_node = RetrievalNode(mock_evidence, mock_track)
    verification_node = VerificationNode()
    response_node = ResponseNode(llm)
    
    # 3. Execute Pipeline
    state = {"query": "Find the person"}
    state = await intent_node.execute(state)
    state = await planner_node.execute(state)
    state = await retrieval_node.execute(state)
    state = await verification_node.execute(state)
    state = await response_node.execute(state)
    
    # 4. Verify Strict Contracts
    assert state["query_intent"] is not None
    assert state["execution_plan"] == ["EvidenceSearchTool"]
    assert state["verified_contract"].verified_count == 2
    assert "abstain_reason" not in state
    assert state["final_response"] == "The person was seen 5 times."

@pytest.mark.asyncio
async def test_full_rag_pipeline_abstain_insufficient_evidence():
    mock_intent = QueryIntent(intent_type=IntentType.SEARCH, entity_type=EntityType.PERSON, is_valid=True, confidence=0.9)
    llm = MockE2ELLMClient(mock_intent, "I apologize, but I don't have enough evidence.")
    
    mock_evidence = AsyncMock(spec=EvidenceService)
    mock_evidence.search_evidence.return_value = [] # Returns NO evidence
    mock_track = AsyncMock(spec=TrackService)
    
    intent_node = IntentNode(llm)
    planner_node = PlannerNode()
    retrieval_node = RetrievalNode(mock_evidence, mock_track)
    verification_node = VerificationNode()
    response_node = ResponseNode(llm)
    
    state = {"query": "Find the person"}
    state = await intent_node.execute(state)
    state = await planner_node.execute(state)
    state = await retrieval_node.execute(state)
    state = await verification_node.execute(state)
    state = await response_node.execute(state)
    
    assert state["verified_contract"] is None
    assert "Insufficient evidence" in state["abstain_reason"]
    assert state["final_response"] == "I apologize, but I don't have enough evidence."
