import pytest
from unittest.mock import AsyncMock, patch

from app.agents.vector.agent import VectorAgent
from app.agents.vector.schemas import VectorResult
from app.schemas.context import VistaContext, ExecutionPlan, UserContext
from app.agents.intent.enums import Intent
from app.agents.intent.schemas import IntentResult

class MockVectorMatch:
    def __init__(self, id, score, camera_id="cam1", description="desc", timestamp=None, bbox=None):
        from datetime import datetime, timezone
        self.id = id
        self.score = score
        self.camera_id = camera_id
        self.description = description
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.bbox = bbox

@pytest.mark.asyncio
async def test_vector_agent_multi_query_fusion():
    # Setup mocked service
    mock_service = AsyncMock()
    # It should be called multiple times for expanded queries. Return different candidates.
    mock_service.search_person.side_effect = [
        [MockVectorMatch(id="doc1", score=0.9)],
        [MockVectorMatch(id="doc2", score=0.85)]
    ]
    
    # Mock expander to return two queries
    mock_expander = AsyncMock()
    mock_expander.expand.return_value = ["red shirt", "red top"]
    
    # Mock encoder
    from unittest.mock import MagicMock
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = [0.1] * 768
    
    agent = VectorAgent(vector_service=mock_service, encoder=mock_encoder)
    agent.expander = mock_expander
    
    # Context
    context = VistaContext(user=UserContext(user_id="1", role="admin"), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["vector_agent"], intent=Intent.PERSON_SEARCH.value)
    context.results["intent_agent"] = IntentResult(
        success=True,
        intent=Intent.PERSON_SEARCH,
        entities={"description": "red shirt"}
    )
    
    result = await agent.execute(context, None)
    
    # Validate that both mock searches contributed to candidates
    assert len(result.person_matches) == 2
    assert result.person_matches[0].id == "doc1"
    assert result.person_matches[1].id == "doc2"
    
    # Ensure evidence bundle mapping succeeded
    assert len(result.evidence) == 2
    
    # Ensure expander was called
    mock_expander.expand.assert_called_once()
