import pytest
from unittest.mock import AsyncMock, patch

from app.agents.vector.expander import QueryExpander
from app.agents.intent.schemas import IntentResult
from app.agents.intent.enums import Intent

@pytest.mark.asyncio
async def test_query_expansion_no_hallucination():
    # Mock LLM Client
    mock_llm = AsyncMock()
    # Mock the return structure expected by QueryExpander
    mock_llm.ainvoke.return_value.content = '{"queries": ["person in red shirt", "person wearing red top"]}'
    
    expander = QueryExpander(llm_client=mock_llm)
    
    intent = IntentResult(
        success=True,
        intent=Intent.PERSON_SEARCH,
        entities={"description": "red shirt"}
    )
    
    expanded = await expander.expand(intent)
    
    # Assert original description is preserved at index 0
    assert expanded[0] == "red shirt"
    # Assert expanded variations are included
    assert "person in red shirt" in expanded
    assert "person wearing red top" in expanded
    
    # Assert LLM was called
    mock_llm.ainvoke.assert_called_once()
