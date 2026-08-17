from app.agents.reasoning.engine.hypothesis_generator import HypothesisGenerator
from app.agents.reasoning.engine.explanation_generator import ExplanationGenerator

class ReasoningService:
    """
    Service layer providing configured instances of components that require external dependencies
    such as the LLM client or Knowledge Graph connections.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def get_hypothesis_generator(self) -> HypothesisGenerator:
        return HypothesisGenerator(llm_client=self.llm_client)
        
    def get_explanation_generator(self) -> ExplanationGenerator:
        return ExplanationGenerator(llm_client=self.llm_client)

    def get_evidence_verifier(self):
        from app.agents.reasoning.engine.evidence_verifier import EvidenceVerifier
        return EvidenceVerifier(llm_client=self.llm_client)
