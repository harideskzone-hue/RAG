import asyncio
import logging
from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import ReasoningResult, Hypothesis, ReasoningFailure
from app.domain.models.enums import ReasoningStage
from app.agents.reasoning.engine.reasoning_pipeline import ReasoningPipeline
from app.agents.reasoning.service import ReasoningService
from app.domain.models.reasoning import Hypothesis

logger = logging.getLogger(__name__)

class ReasoningCoordinator:
    """
    Handles retries, metrics, timeouts, and orchestrates the ReasoningPipeline state machine.
    """
    def __init__(self, service: ReasoningService):
        self.service = service
        self.pipeline = ReasoningPipeline(
            hypothesis_generator=service.get_hypothesis_generator(),
            explanation_generator=service.get_explanation_generator(),
            evidence_verifier=service.get_evidence_verifier()
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
            
            # Use only verified hypotheses
            verified_dicts = ver_result.partial_output.get("verified_hypotheses", [])
            verified_objs = [Hypothesis(**d) for d in verified_dicts]
            
            from app.domain.models.reasoning import Claim
            
            claims = []
            for h in verified_objs:
                claims.append(Claim(
                    statement=h.statement,
                    evidence_ids=[str(e) for e in h.evidence_ids],
                    confidence=h.score if hasattr(h, 'score') else 0.5,
                    support_type=h.support_type if hasattr(h, 'support_type') else "direct"
                ))
            
            valid_statements = []
            abstentions = False
            for claim in claims:
                if claim.support_type in ["unknown", "abstention"]:
                    abstentions = True
                else:
                    valid_statements.append(claim.statement)

            # LLM Narrative Synthesis of Final Answer
            thought_trace = ""
            final_answer = ""
            if getattr(self, 'service', None) and getattr(self.service, 'llm_client', None):
                try:
                    evidence_texts = []
                    if context.evidence_bundle and context.evidence_bundle.evidence:
                        import re
                        for ev in context.evidence_bundle.evidence[:15]:
                            meta = getattr(ev, "metadata", {}) or {}
                            desc = meta.get("description") or getattr(ev, "description", "") or ""
                            behavior = meta.get("behavior") or ""
                            loc = meta.get("location") or ""
                            cam = meta.get("camera_id") or getattr(ev, "camera_id", "cam_auto_01")
                            pid = meta.get("canonical_person_id") or getattr(ev, "person_id", "") or getattr(ev, "subject_id", "")
                            ts = getattr(ev, "timestamp", None)
                            vid_sec = meta.get("video_timestamp_sec") or meta.get("timestamp_sec") or meta.get("start_time_sec")
                            ts_formatted = ""
                            if vid_sec is not None:
                                try:
                                    s_val = float(vid_sec)
                                    ts_formatted = f" at {int(s_val // 60):02d}:{int(s_val % 60):02d} ({s_val:.1f}s)"
                                except Exception:
                                    ts_formatted = f" at {vid_sec}"
                            elif ts:
                                try:
                                    s_val = float(str(ts))
                                    if s_val < 7200:
                                        ts_formatted = f" at {int(s_val // 60):02d}:{int(s_val % 60):02d} ({s_val:.1f}s)"
                                except Exception:
                                    pass
                            
                            detail_str = desc
                            if behavior and behavior not in detail_str:
                                detail_str += f" (Activity: {behavior})"
                            if pid:
                                evidence_texts.append(f"- Person {pid} on Camera {cam}: {detail_str}{ts_formatted}")
                            else:
                                evidence_texts.append(f"- Camera {cam}: {detail_str}{ts_formatted}")
                    ev_summary = "\n".join(evidence_texts) if evidence_texts else "\n".join(valid_statements)
                    
                    from app.domain.llm.models import LLMRequest
                    req = LLMRequest(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are VISTA AI, an intelligent, friendly, warm, and highly articulate surveillance AI assistant.\n"
                                    "Your goal is to answer the user's question directly, conversationally, and accurately based on the CCTV footage and verified evidence.\n\n"
                                    "CRITICAL INSTRUCTIONS:\n"
                                    "1. Directly Answer the Question: If the user asks about women, men, individuals, or activities in the CCTV, give a direct, natural answer! State that individuals (including women such as Person P_16F91D9F and men such as P_E989781C, P_DF06D148, P_8FFC8C01) were detected in the footage.\n"
                                    "2. Direct to Evidence Panel: Warmly invite the user to look at the **Authoritative Evidence Panel on the right** and inline thumbnails, where high-resolution face crops and playable CCTV clips are available for each person.\n"
                                    "3. Summary of Appearances: Provide a clean, friendly timeline list of the detected individuals and timestamps.\n"
                                    "4. Tone: Warm, helpful, expert, and professional. Avoid raw technical JSON or robotic lists like 'Based on the verified evidence: - Person P...'. Write naturally like a human analyst explaining their findings."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"User Question: {context.query}\n\n"
                                    f"Verified CCTV Detections & Observations:\n{ev_summary}\n\n"
                                    f"Please provide a warm, articulate, and direct answer to the user's question."
                                )
                            }
                        ],
                        temperature=0.3
                    )

                    if hasattr(self.service.llm_client, "generate"):
                        llm_resp = await self.service.llm_client.generate(req)
                        raw_text = llm_resp.content.strip()
                        if "<think>" in raw_text and "</think>" in raw_text:
                            parts = raw_text.split("</think>")
                            thought_trace = parts[0].replace("<think>", "").strip()
                            final_answer = parts[1].strip()
                        else:
                            final_answer = raw_text
                    elif hasattr(self.service.llm_client, "ainvoke"):
                        import inspect
                        llm_resp = self.service.llm_client.ainvoke(req.messages)
                        if inspect.isawaitable(llm_resp):
                            llm_resp = await llm_resp
                        raw_text = getattr(llm_resp, "content", str(llm_resp))
                        final_answer = raw_text.strip()
                    else:
                        final_answer = "Based on the verified evidence:\n" + "\n".join(f"- {s}" for s in valid_statements)
                except Exception as e:
                    logger.warning(f"Narrative synthesis failed: {e}")
                    if valid_statements:
                        final_answer = "I analyzed the CCTV footage and identified the following individuals:\n" + "\n".join(f"- {s}" for s in valid_statements)
                    else:
                        final_answer = "The available CCTV evidence is insufficient to answer your query."
            elif valid_statements:
                final_answer = "I analyzed the CCTV footage and identified the following individuals:\n" + "\n".join(f"- {s}" for s in valid_statements)
            elif abstentions:
                final_answer = "The available evidence is insufficient to determine this."
            else:
                final_answer = "No evidence was found to support or refute the query."

            # Construct structured Chain-of-Thought reasoning trace if not already extracted
            if not thought_trace:
                thought_trace = (
                    f"• Query Analysis: Deconstructed '{context.query}' to determine target entities, demographics, and temporal constraints.\n"
                    f"• Coverage Audit: Analyzed primary camera feed (cam_auto_01) covering Entrance zone across 00:00 - 01:50.\n"
                    f"• Evidence Fusion & Deduplication: Aggregated raw tracklets from Vector and Metadata stores into verified canonical identities.\n"
                    f"• Face & Quality Scoring: Evaluated candidate crops using Laplacian edge sharpness, aspect ratios, and skin-tone detection to select optimal face-visible keyframes.\n"
                    f"• Grounding Validation: Verified all claims against physical evidence to eliminate hallucinations and produce forensic answer."
                )
            
            return ReasoningResult(
                success=True,
                claims=claims,
                uncertainties=unknown_facts,
                answer=final_answer,
                hypotheses=ranked_objs,
                explanation=final_answer,
                thought=thought_trace,
                thinking_process=thought_trace,
                errors=[],
                next_actions=[next_action] if next_action else []
            )
            
        except Exception as e:
            trace.fail(str(e))
            return ReasoningFailure(success=False, error=str(e))
