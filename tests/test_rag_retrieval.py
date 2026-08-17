import pytest
from unittest.mock import AsyncMock

from app.schemas.intent import QueryIntent, IntentType, EntityType
from app.graph.nodes.planner import PlannerNode
from app.graph.nodes.retrieval import RetrievalNode
from app.services.db_services import EvidenceService, TrackService

@pytest.mark.asyncio
async def test_planner_deterministic_routing():
    planner = PlannerNode()
    
    # 1. Search without identity -> EvidenceSearchTool
    intent1 = QueryIntent(intent_type=IntentType.SEARCH, entity_type=EntityType.PERSON)
    state1 = {"query_intent": intent1}
    res1 = await planner.execute(state1)
    assert res1["execution_plan"] == ["EvidenceSearchTool"]
    
    # 2. Search with identity -> PersonSearchTool
    intent2 = QueryIntent(intent_type=IntentType.SEARCH, entity_type=EntityType.PERSON, identity_target="P123")
    state2 = {"query_intent": intent2}
    res2 = await planner.execute(state2)
    assert res2["execution_plan"] == ["PersonSearchTool"]

@pytest.mark.asyncio
async def test_retrieval_node_execution():
    mock_evidence = AsyncMock(spec=EvidenceService)
    mock_track = AsyncMock(spec=TrackService)
    
    mock_evidence.search_evidence.return_value = [{"evidence_id": "e1"}]
    
    retrieval = RetrievalNode(mock_evidence, mock_track)
    
    intent = QueryIntent(intent_type=IntentType.SEARCH, entity_type=EntityType.PERSON)
    state = {
        "query_intent": intent,
        "execution_plan": ["EvidenceSearchTool"]
    }
    
    res = await retrieval.execute(state)
    assert len(res["retrieved_evidence"]) == 1
    assert res["retrieved_evidence"][0]["tool"] == "EvidenceSearchTool"
    mock_evidence.search_evidence.assert_called_once()
