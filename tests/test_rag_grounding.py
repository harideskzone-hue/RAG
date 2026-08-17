import pytest
from app.graph.nodes.verification import VerifiedResultContract
from app.graph.nodes.grounding import GroundingValidatorNode

@pytest.mark.asyncio
async def test_grounding_validator_accepts_valid_response():
    validator = GroundingValidatorNode()
    
    contract = VerifiedResultContract(
        verified_count=5,
        verified_persons=["P1234"],
        cameras=["CAM_01"]
    )
    
    state = {
        "verified_contract": contract,
        "final_response": "I found 5 people. Specifically person P1234 on camera CAM_01."
    }
    
    res = await validator.execute(state)
    assert res["grounding_valid"] is True
    assert res["final_response"] == state["final_response"]

@pytest.mark.asyncio
async def test_grounding_validator_rejects_hallucinated_counts():
    validator = GroundingValidatorNode()
    
    contract = VerifiedResultContract(
        verified_count=5,
    )
    
    # LLM hallucinates 999
    state = {
        "verified_contract": contract,
        "final_response": "I found 999 people."
    }
    
    res = await validator.execute(state)
    assert res["grounding_valid"] is False
    assert "cannot safely answer" in res["final_response"]
    assert "999" in res["abstain_reason"]

@pytest.mark.asyncio
async def test_grounding_validator_rejects_hallucinated_ids():
    validator = GroundingValidatorNode()
    
    contract = VerifiedResultContract(
        verified_persons=["P1234"]
    )
    
    # LLM hallucinates P9999
    state = {
        "verified_contract": contract,
        "final_response": "The person is P9999."
    }
    
    res = await validator.execute(state)
    assert res["grounding_valid"] is False
    assert "P9999" in res["abstain_reason"]
