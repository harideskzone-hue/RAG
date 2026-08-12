from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult
from app.domain.models.enums import ReasoningStage
from app.agents.reasoning.engine.correlator import Correlator
from app.agents.reasoning.engine.contradiction_detector import ContradictionDetector
from app.agents.reasoning.engine.gap_analyzer import GapAnalyzer
from app.agents.reasoning.engine.hypothesis_generator import HypothesisGenerator
from app.agents.reasoning.engine.hypothesis_ranker import HypothesisRanker
from app.agents.reasoning.engine.explanation_generator import ExplanationGenerator
from app.agents.reasoning.engine.evidence_verifier import EvidenceVerifier

class ReasoningPipeline:
    """
    Executes the deterministic and semantic reasoning stages sequentially.
    """
    def __init__(self, hypothesis_generator: HypothesisGenerator, explanation_generator: ExplanationGenerator):
        self.correlator = Correlator()
        self.contradiction_detector = ContradictionDetector()
        self.gap_analyzer = GapAnalyzer()
        self.hypothesis_generator = hypothesis_generator
        self.hypothesis_ranker = HypothesisRanker()
        self.explanation_generator = explanation_generator
        self.evidence_verifier = EvidenceVerifier()
        
    async def run_stage(self, context: ReasoningContext, stage: ReasoningStage, **kwargs) -> EngineResult:
        if stage == ReasoningStage.CORRELATION:
            return self.correlator.run(context)
        elif stage == ReasoningStage.CONTRADICTION:
            # We don't actually run it concurrently here if we rely on coordinator to pass state explicitly,
            # but we can return it. Wait, the Coordinator calls run_stage for each.
            # To do it concurrently, we'd need to change Coordinator. Let's leave run_stage as is,
            # and modify ReasoningCoordinator to gather them instead.
            return self.contradiction_detector.run(context)
        elif stage == ReasoningStage.GAP_ANALYSIS:
            return self.gap_analyzer.run(context)
        elif stage == ReasoningStage.HYPOTHESIS_GENERATION:
            return await self.hypothesis_generator.run(context, kwargs.get("correlations", []), kwargs.get("gaps", []))
        elif stage == ReasoningStage.HYPOTHESIS_RANKING:
            return self.hypothesis_ranker.run(context, kwargs.get("hypotheses", []))
        elif stage == ReasoningStage.EXPLANATION:
            return await self.explanation_generator.run(context, kwargs.get("ranked_hypotheses", []))
        elif stage == ReasoningStage.VERIFICATION:
            return self.evidence_verifier.run(context, kwargs.get("explanation", ""), kwargs.get("hypotheses", []))
        
        return EngineResult(success=False, errors=[f"Unknown stage {stage}"])
