import pytest
from unittest.mock import AsyncMock

from app.schemas.intent import QueryIntent, IntentType, EntityType
from app.graph.nodes.response import ResponseNode
from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMRequest, LLMResponse
from app.graph.nodes.verification import VerifiedResultContract

class AdversarialLLMClient(BaseLLMClient):
    def __init__(self, expected_response):
        self.expected_response = expected_response
        self.system_prompt_received = None
        self.user_prompt_received = None
        
    def capabilities(self): pass
    
    async def generate(self, request: LLMRequest, **kwargs):
        self.system_prompt_received = request.messages[0]["content"]
        self.user_prompt_received = request.messages[1]["content"]
        return LLMResponse(content=self.expected_response)
        
    async def generate_structured(self, request: LLMRequest, schema: type, **kwargs):
        pass

@pytest.mark.asyncio
async def test_adversarial_no_keyword_shortcuts():
    """
    Test that the ResponseNode relies entirely on the LLM client,
    and doesn't contain hardcoded shortcuts like 'if query == "how many": return 5'.
    """
    # LLM will return a specific string regardless of the query, proving that
    # the Node passes it through rather than intercepting it with regex.
    mock_llm = AdversarialLLMClient("The LLM semantic response.")
    node = ResponseNode(mock_llm)
    
    queries = [
        "how many people are there?",
        "where did the red shirt go?",
        "count the vehicles"
    ]
    
    contract = VerifiedResultContract(verified_count=5)
    
    for q in queries:
        state = {"query": q, "verified_contract": contract}
        res = await node.execute(state)
        
        # 1. The response must come from the LLM, not a hardcoded template
        assert res["final_response"] == "The LLM semantic response."
        
        # 2. The LLM must be given the verified contract to ground its answer
        assert "verified_count\": 5" in mock_llm.user_prompt_received
        
        # 3. The LLM must be explicitly told NOT to invent facts
        assert "MUST NOT invent, add, or alter any evidence" in mock_llm.system_prompt_received

@pytest.mark.asyncio
async def test_model_swap_independence():
    """
    Proves answer generation remains model-independent and only relies on BaseLLMClient.
    """
    # Swap out Groq for "Ollama" Mock
    class OllamaMockClient(BaseLLMClient):
        def capabilities(self): pass
        async def generate(self, request, **kwargs):
            return LLMResponse(content="Ollama model response.")
        async def generate_structured(self, request, schema, **kwargs): pass
            
    node = ResponseNode(OllamaMockClient())
    state = {"query": "test", "verified_contract": VerifiedResultContract()}
    res = await node.execute(state)
    
    # Prove the Node successfully used the swapped model via BaseLLMClient
    assert res["final_response"] == "Ollama model response."
