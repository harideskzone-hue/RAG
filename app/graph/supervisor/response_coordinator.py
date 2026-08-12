from typing import Any

from app.schemas.context import VistaContext


class ResponseCoordinator:
    """
    Assembles Evidence, Confidence, Citations, and Results into a final response.
    Separates orchestration from presentation.
    """
    def generate_response(self, context: VistaContext) -> dict[str, Any]:
        print(f"EVIDENCE BUNDLE: {context.evidence_bundle}")
        print(f"RESULTS: {context.results.keys()}")
        print(f"UNAUTHORIZED: {context.results.get('unauthorized_evidence_found')}")

        # Extract evidence from bundle if present
        bundle_evidence = []
        if context.evidence_bundle and context.evidence_bundle.evidence:
            bundle_evidence = context.evidence_bundle.evidence
            
        # Check if NO_AUTHORIZED_EVIDENCE
        if not bundle_evidence and getattr(context.user, "allowed_cameras", None):
            return {
                "status": "NO_AUTHORIZED_EVIDENCE",
                "error": None,
                "final_answer": "No authorized evidence found for this query.",
                "evidence": [],
                "citations": [],
                "overall_confidence": 0.0,
                "agent_decisions": context.agent_decisions
            }

        # Determine status based on whether error results exist
        has_error = "error" in context.results
        status = "error" if has_error else "success"

        # Check intent result first for conversational fallbacks
        if "intent_agent" in context.results:
            intent_result = context.results["intent_agent"]
            intent_val = getattr(intent_result, "intent", None)
            if intent_val and str(intent_val.value if hasattr(intent_val, "value") else intent_val).lower() == "greeting":
                final_answer = "Hi! I'm VISTA AI."
                status = "success"
                return {
                    "status": status,
                    "error": None,
                    "final_answer": final_answer,
                    "evidence": [],
                    "citations": [],
                    "overall_confidence": 1.0,
                    "agent_decisions": context.agent_decisions
                }
            elif getattr(intent_result, "requires_clarification", False) or (intent_val and str(intent_val.value if hasattr(intent_val, "value") else intent_val).lower() == "unknown"):
                final_answer = "The available CCTV evidence is insufficient to answer your query. Please clarify."
                status = "success"
                return {
                    "status": status,
                    "error": None,
                    "final_answer": final_answer,
                    "evidence": [],
                    "citations": [],
                    "overall_confidence": 0.0,
                    "agent_decisions": context.agent_decisions
                }

        # Check for error in results first
        if has_error:
            final_answer = context.results.get("error", "Unknown error")
            status = "error"
        # Check for guardrail block
        elif "guardrail_agent" in context.results:
            guardrail_result = context.results.get("guardrail_agent")
            if guardrail_result and not getattr(guardrail_result, "is_safe", True):
                final_answer = "Response blocked by safety guardrails: " + ", ".join(getattr(guardrail_result, "violations", []))
                status = "error"
            else:
                # Guardrail passed, check reasoning
                reasoning_result = context.results.get("reasoning_agent")
                if reasoning_result:
                    if hasattr(reasoning_result, "status") and getattr(reasoning_result.status, "value", str(reasoning_result.status)) == "error":
                        # Try to get error details from reasoning result, fallback to empty string
                        metadata = getattr(reasoning_result, "metadata", None)
                        if isinstance(metadata, dict):
                            errors = metadata.get("errors", [])
                            final_answer = "error: " + "; ".join(errors) if errors else ""
                        else:
                            errors = getattr(reasoning_result, "errors", [])
                            final_answer = "error: " + "; ".join(errors) if errors else ""
                    else:
                        # Try to get answer from metadata first
                        metadata = getattr(reasoning_result, "metadata", None)
                        if isinstance(metadata, dict):
                            final_answer = metadata.get("answer") or metadata.get("explanation", "")
                        else:
                            final_answer = ""
                        # Fallback to direct explanation attribute
                        if not final_answer and hasattr(reasoning_result, "explanation"):
                            final_answer = getattr(reasoning_result, "explanation", "")
                else:
                    # No reasoning agent result but it was required
                    if context.execution_plan and "reasoning_agent" in context.execution_plan.agents:
                        final_answer = "Response blocked: Reasoning agent failed to execute."
                        status = "error"
                    else:
                        final_answer = ""
                        status = "error"
        else:
            # No guardrail agent result, check reasoning
            reasoning_result = context.results.get("reasoning_agent")
            if reasoning_result:
                if hasattr(reasoning_result, "status") and getattr(reasoning_result.status, "value", str(reasoning_result.status)) == "error":
                    metadata = getattr(reasoning_result, "metadata", None)
                    if isinstance(metadata, dict):
                        errors = metadata.get("errors", [])
                        final_answer = "Response blocked: " + "; ".join(errors) if errors else "Response blocked by reasoning parser/safety."
                    else:
                        errors = getattr(reasoning_result, "errors", [])
                        final_answer = "Response blocked: " + "; ".join(errors) if errors else "Response blocked by reasoning parser/safety."
                    status = "error"
                else:
                    # Try to get answer from metadata first
                    metadata = getattr(reasoning_result, "metadata", None)
                    if isinstance(metadata, dict):
                        final_answer = metadata.get("answer") or metadata.get("explanation", "")
                    else:
                        final_answer = ""
                    # Fallback to direct explanation attribute
                    if not final_answer and hasattr(reasoning_result, "answer"):
                        final_answer = getattr(reasoning_result, "answer", "")
                    if not final_answer and hasattr(reasoning_result, "explanation"):
                        final_answer = getattr(reasoning_result, "explanation", "")
                    
                    if not final_answer:
                        final_answer = "Response blocked: Model failed to provide a conclusive answer."
                        status = "error"
            else:
                # No reasoning agent result
                if context.execution_plan and "reasoning_agent" in context.execution_plan.agents:
                    final_answer = "Response blocked: Reasoning agent failed to execute."
                    status = "error"
                elif "report_agent" in context.results:
                    final_answer = "Report generated successfully."
                    status = "success"
                elif context.results:
                    final_answer = "Pipeline executed successfully."
                    status = "success"
                else:
                    final_answer = "Response blocked: Reasoning agent failed to execute."
                    status = "error"

        formatted_evidence = []
        if status != "error":
            # Extract explicitly cited evidence IDs from reasoning claims
            reasoning_result = context.results.get("reasoning_agent")
            cited_evidence_ids = set()
            
            if reasoning_result:
                metadata = getattr(reasoning_result, "metadata", {})
                claims = metadata.get("claims", [])
                if not claims and hasattr(reasoning_result, "claims"):
                    claims = getattr(reasoning_result, "claims", [])
                    
                if hasattr(claims, "__iter__") and not isinstance(claims, str):
                    try:
                        for claim in claims:
                            cited = claim.get("evidence_ids", []) if isinstance(claim, dict) else getattr(claim, "evidence_ids", [])
                            if cited and hasattr(cited, "__iter__") and not isinstance(cited, str):
                                cited_evidence_ids.update(str(c) for c in cited)
                    except Exception:
                        pass
            
            # Map valid citations against the EvidenceBundle
            if context.evidence_bundle:
                for ev in context.evidence_bundle.evidence:
                    if str(ev.evidence_id) in cited_evidence_ids:
                        formatted_evidence.append({
                            "evidence_id": str(ev.evidence_id),
                            "source": ev.source,
                            "camera_id": ev.metadata.get("camera_id"),
                            "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
                            "description": ev.metadata.get("description"),
                            "confidence": ev.confidence
                        })

        return {
            "status": status,
            "error": context.results.get("error"),
            "final_answer": final_answer,
            "evidence": formatted_evidence,
            "citations": [cit.model_dump() for cit in context.citations],
            "overall_confidence": context.confidence_score,
            "agent_decisions": context.agent_decisions
        }