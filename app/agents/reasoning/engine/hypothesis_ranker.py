from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult, Hypothesis

class HypothesisRanker:
    """
    Deterministic/Heuristic component.
    Scores hypotheses based on probabilistic weighting and evidence density.
    """
    def run(self, context: ReasoningContext, hypotheses_dicts: list[dict]) -> EngineResult:
        ranked = []
        
        for h_dict in hypotheses_dicts:
            try:
                # Handle both Hypothesis and new Claim schema formats
                if "statement" in h_dict:
                    h = Hypothesis(
                        statement=h_dict["statement"],
                        evidence_ids=h_dict.get("evidence_ids", []),
                        support_type=h_dict.get("support_type", "direct")
                    )
                else:
                    h = Hypothesis(**h_dict)
                
                # Retrieve actual confidence weights from the graph/blackboard
                evidence_bonus = 0.0
                if getattr(context, "relationships", None) is not None:
                    all_rels = {r.relationship_id: r for r in context.relationships}
                    for ev_id in h.evidence_ids:
                        if ev_id in all_rels:
                            conf = all_rels[ev_id].confidence
                            evidence_bonus += (conf * 0.1)
                            h.evidence_weights[str(ev_id)] = conf
                        else:
                            evidence_bonus += 0.05 # Default if missing
                else:
                    evidence_bonus = len(h.evidence_ids) * 0.1
                
                # Simple scoring heuristic
                base_score = 0.5
                gap_penalty = len(h.missing_evidence) * 0.05
                contradiction_penalty = len(h.contradicting_evidence) * 0.2
                
                h.score = max(0.0, min(1.0, base_score + evidence_bonus - gap_penalty - contradiction_penalty))
                ranked.append(h)
            except Exception:
                pass
                
        # Sort by score descending
        ranked.sort(key=lambda x: x.score, reverse=True)
        
        return EngineResult(
            success=True,
            partial_output={"ranked_hypotheses": [h.model_dump() for h in ranked]}
        )
