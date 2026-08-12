import pytest
from app.agents.reasoning.agent import ReasoningAgent
from app.schemas.context import VistaContext
from app.domain.models.reasoning import ReasoningResult, Claim
from app.domain.models.agent_result import AgentResult

@pytest.mark.asyncio
async def test_confidence_propagation():
    agent = ReasoningAgent()
    
    # We will mock the coordinator to return a ReasoningResult with specific claims
    class MockCoordinator:
        async def execute(self, context):
            return ReasoningResult(
                success=True,
                claims=[
                    Claim(statement="A", evidence_ids=["1"], confidence=0.42, support_type="direct"),
                    Claim(statement="B", evidence_ids=["2"], confidence=0.42, support_type="direct")
                ],
                uncertainties=[],
                answer="Mock answer",
                hypotheses=[],
                explanation="Mock explanation",
                errors=[],
                next_actions=[]
            )
            
    agent.coordinator = MockCoordinator()
    
    from app.domain.evidence import EvidenceBundle, BaseEvidence
    import datetime
    from uuid import uuid4
    
    from app.schemas.context import UserContext
    context = VistaContext(conversation_id="test", current_query="test query", user=UserContext(user_id="test", role="admin"))
    context.evidence_bundle = EvidenceBundle(
        evidence=[BaseEvidence(evidence_id=uuid4(), source="test", timestamp=datetime.datetime.now(datetime.timezone.utc), confidence=1.0, metadata={})]
    )
    
    # Execute agent
    result: AgentResult = await agent.execute(context)
    
    # Verify overall confidence is 0.42
    assert result.confidence.overall == 0.42

@pytest.mark.asyncio
async def test_confidence_aggregation():
    agent = ReasoningAgent()
    
    # We will mock the coordinator to return multiple claims
    class MockCoordinator:
        async def execute(self, context):
            return ReasoningResult(
                success=True,
                claims=[
                    Claim(statement="A", evidence_ids=["1"], confidence=0.42, support_type="direct"),
                    Claim(statement="B", evidence_ids=["2"], confidence=0.71, support_type="direct"),
                    Claim(statement="C", evidence_ids=["3"], confidence=0.93, support_type="direct")
                ],
                uncertainties=[],
                answer="Mock answer",
                hypotheses=[],
                explanation="Mock explanation",
                errors=[],
                next_actions=[]
            )
            
    agent.coordinator = MockCoordinator()
    
    from app.domain.evidence import EvidenceBundle, BaseEvidence
    import datetime
    from uuid import uuid4
            
    from app.schemas.context import UserContext
    context = VistaContext(conversation_id="test", current_query="test query", user=UserContext(user_id="test", role="admin"))
    context.evidence_bundle = EvidenceBundle(
        evidence=[BaseEvidence(evidence_id=uuid4(), source="test", timestamp=datetime.datetime.now(datetime.timezone.utc), confidence=1.0, metadata={})]
    )
    
    # Execute agent
    result: AgentResult = await agent.execute(context)
    
    # (0.42 + 0.71 + 0.93) / 3 = 2.06 / 3 = 0.68666...
    assert round(result.confidence.overall, 2) == 0.69
