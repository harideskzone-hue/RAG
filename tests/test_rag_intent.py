import pytest
from unittest.mock import AsyncMock

from app.schemas.intent import QueryIntent, IntentType, EntityType
from app.graph.nodes.intent import IntentNode
from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMRequest

class MockLLMClient(BaseLLMClient):
    def __init__(self, mock_intent: QueryIntent):
        self.mock_intent = mock_intent
        
    def capabilities(self):
        pass
        
    async def generate(self, request, **kwargs):
        pass
        
    async def generate_structured(self, request: LLMRequest, schema: type, **kwargs):
        # We simulate the LLM returning the correctly structured output
        return self.mock_intent

@pytest.mark.asyncio
async def test_intent_parsing_success():
    # Simulate a query where LLM confidently parsed it
    mock_intent = QueryIntent(
        intent_type=IntentType.SEARCH,
        entity_type=EntityType.PERSON,
        confidence=0.9,
        is_valid=True
    )
    mock_client = MockLLMClient(mock_intent)
    node = IntentNode(mock_client)
    
    state = {"query": "Show me the person in red"}
    result = await node.execute(state)
    
    assert result["query_intent"].intent_type == IntentType.SEARCH
    assert "abstain_reason" not in result

@pytest.mark.asyncio
async def test_intent_parsing_abstain_on_low_confidence():
    mock_intent = QueryIntent(
        intent_type=IntentType.UNKNOWN,
        entity_type=EntityType.UNKNOWN,
        confidence=0.3, # Too low
        is_valid=True
    )
    mock_client = MockLLMClient(mock_intent)
    node = IntentNode(mock_client)
    
    state = {"query": "idk"}
    result = await node.execute(state)
    
    assert result["query_intent"] is None
    assert "abstain_reason" in result
    assert "confidently determine" in result["abstain_reason"]
