import asyncio
from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import ReasoningResult, Hypothesis, ReasoningFailure
from app.domain.models.enums import ReasoningStage
from app.agents.reasoning.engine.reasoning_pipeline import ReasoningPipeline
from app.agents.reasoning.service import ReasoningService
from app.domain.models.reasoning import Hypothesis

class ReasoningCoordinator:
    """
    Handles retries, metrics, timeouts, and orchestrates the ReasoningPipeline state machine.
    """
    def __init__(self, service: ReasoningService):
        self.pipeline = ReasoningPipeline(
            hypothesis_generator=service.get_hypothesis_generator(),
            explanation_generator=service.get_explanation_generator()
        )
        
    async def execute(self, context: ReasoningContext) -> ReasoningResult | ReasoningFailure:
        trace = context.trace
        
        try:
            # 1. Correlation
            trace.transition(ReasoningStage.CORRELATION)
            corr_result = await self.pipeline.run_stage(context, trace.current_stage)
            correlations = corr_result.partial_output.get("new_correlations", 0)
            
            # 2. Contradictions & 3. Gap Analysis (Concurrent DAG execution)
            trace.transition(ReasoningStage.CONTRADICTION) # Technically both are running now
            
            # Create tasks for gather
            contra_task = self.pipeline.run_stage(context, ReasoningStage.CONTRADICTION)
            gap_task = self.pipeline.run_stage(context, ReasoningStage.GAP_ANALYSIS)
            
            contra_result, gap_result = await asyncio.gather(contra_task, gap_task)
            
            trace.transition(ReasoningStage.GAP_ANALYSIS) # Log completion
            
            gaps = gap_result.partial_output.get("gaps", [])
            next_action = gap_result.next_action
            
            # 4. Hypothesis Generation
            trace.transition(ReasoningStage.HYPOTHESIS_GENERATION)
            gen_result = await self.pipeline.run_stage(context, trace.current_stage, correlations=correlations, gaps=gaps)
            if not gen_result.success:
                trace.fail(gen_result.errors[0])
                return ReasoningFailure(success=False, error=gen_result.errors[0])
                
            raw_hypotheses = gen_result.partial_output.get("hypotheses", [])
            # HypothesisGenerator.run stores its own answer as 'explanation'
            explanation = gen_result.partial_output.get("explanation", "")
            unknown_facts = gen_result.partial_output.get("unknown_facts", [])
            
            # 5. Hypothesis Ranking
            trace.transition(ReasoningStage.HYPOTHESIS_RANKING)
            rank_result = await self.pipeline.run_stage(context, trace.current_stage, hypotheses=raw_hypotheses)
            if not rank_result.success:
                trace.fail(rank_result.errors[0])
                return ReasoningFailure(success=False, error=rank_result.errors[0])
                
            ranked_objs = rank_result.partial_output.get("ranked_hypotheses", [])
            
            # 6. Explanation Generation (Bypassed)
            # The semantic explanation is already generated directly by HypothesisGenerator
            
            # 7. Verification
            trace.transition(ReasoningStage.VERIFICATION)
            ranked_objs = [Hypothesis(**d) for d in ranked_objs]
            ver_result = await self.pipeline.run_stage(context, trace.current_stage, explanation=explanation, hypotheses=ranked_objs)
            if not ver_result.success:
                trace.fail(ver_result.errors[0])
                return ReasoningFailure(success=False, error=ver_result.errors[0])
                
            trace.transition(ReasoningStage.COMPLETED)
            
            from app.domain.models.reasoning import Claim
            
            claims = []
            for h in ranked_objs:
                claims.append(Claim(
                    statement=h.statement,
                    evidence_ids=[str(e) for e in h.evidence_ids],
                    confidence=h.score if hasattr(h, 'score') else 0.5,
                    support_type=h.support_type if hasattr(h, 'support_type') else "direct"
                ))
            
            # Deterministic Answer Construction
            valid_statements = []
            abstentions = False
            for claim in claims:
                if claim.support_type in ["unknown", "abstention"]:
                    abstentions = True
                else:
                    valid_statements.append(claim.statement)
                    
            if valid_statements:
                final_answer = "Based on the verified evidence:\n" + "\n".join(f"- {s}" for s in valid_statements)
            elif abstentions:
                final_answer = "The available evidence is insufficient to determine this."
            else:
                final_answer = "No evidence was found to support or refute the query."
            
            return ReasoningResult(
                success=True,
                claims=claims,
                uncertainties=unknown_facts,
                answer=final_answer,
                hypotheses=ranked_objs,
                explanation=final_answer,
                errors=[],
                next_actions=[next_action] if next_action else []
            )
            
        except Exception as e:
            trace.fail(str(e))
            return ReasoningFailure(success=False, error=str(e))
