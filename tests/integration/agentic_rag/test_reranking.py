import pytest
from unittest.mock import AsyncMock

from app.agents.vector.agent import VectorAgent
from app.agents.vector.reranker import PassThroughReranker
from app.schemas.context import VistaContext, ExecutionPlan, UserContext
from app.agents.intent.enums import Intent

class MockMatch:
    def __init__(self, score, id="mock"):
        self.score = score
        self.id = id
        from datetime import datetime, timezone
        self.timestamp = datetime.now(timezone.utc)
        self.camera_id = "cam"
        self.description = "desc"
        self.bbox = None
        self.license_plate = None

@pytest.mark.asyncio
async def test_vector_agent_reranking():
    mock_service = AsyncMock()
    # Return out of order
    mock_service.search_person.return_value = [MockMatch(score=0.5, id="mock1"), MockMatch(score=0.9, id="mock2")]
    
    from unittest.mock import MagicMock
    agent = VectorAgent(vector_service=mock_service, encoder=MagicMock())
    # Ensure it's using the pass-through reranker
    assert isinstance(agent.reranker, PassThroughReranker)
    
    context = VistaContext(user=UserContext(user_id="1", role="admin"), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["vector_agent"], intent=Intent.PERSON_SEARCH.value)
    
    result = await agent.execute(context, None)
    
    # Validation: PassThroughReranker sorts by score descending
    assert len(result.person_matches) == 2
    assert result.person_matches[0].score == 0.9
    assert result.person_matches[1].score == 0.5
