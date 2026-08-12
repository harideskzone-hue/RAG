from app.agents.intent.enums import Intent
from app.domain.models.confidence import (
    ConfidenceExplanation,
    ConfidencePolicy,
    ConfidenceReport,
    ConfidenceResult,
)
from app.domain.evidence import EvidenceBundle


class ConfidenceEngine:
    """
    Evaluates confidence across multiple dimensions (Source, Retrieval, Temporal, Agreement, Completeness, Readiness).
    """
    def __init__(self, policy: ConfidencePolicy | None = None):
        self.policy = policy or ConfidencePolicy()

    def evaluate(self, bundle: EvidenceBundle, intent: str) -> ConfidenceResult:
        explanations: list[ConfidenceExplanation] = []
        
        # 1. Source Confidence & 2. Retrieval Confidence
        meta_conf, vec_conf, vid_conf = self._evaluate_sources(bundle, explanations)
        
        # 3. Cross-Source Agreement
        agreement_conf = self._evaluate_agreement(bundle, explanations)
        
        # 4. Temporal Consistency
        temporal_conf = self._evaluate_temporal(bundle, explanations)
        
        # 5. Evidence Completeness
        completeness_conf = self._evaluate_completeness(bundle, intent, explanations)
        
        # Calculate overall weighted score
        weights = {"source": 0.4, "temporal": 0.2, "completeness": 0.2, "agreement": 0.2}
        
        source_avg = 0.0
        active_sources = 0
        if meta_conf is not None:
            source_avg += meta_conf
            active_sources += 1
        if vec_conf is not None:
            source_avg += vec_conf
            active_sources += 1
        if vid_conf is not None:
            source_avg += vid_conf
            active_sources += 1
            
        if active_sources > 0:
            source_avg /= active_sources
            
        overall_score = (
            (source_avg * weights["source"]) +
            (temporal_conf * weights["temporal"]) +
            (completeness_conf * weights["completeness"]) +
            (agreement_conf * weights["agreement"])
        )
        
        # 6. Reasoning Readiness & Action Recommendation
        reasoning_ready, next_action, needs_clarification = self._determine_next_action(overall_score, completeness_conf, intent)
        
        report = ConfidenceReport(
            overall=overall_score,
            metadata=meta_conf,
            vector=vec_conf,
            video=vid_conf,
            temporal=temporal_conf,
            completeness=completeness_conf,
            agreement=agreement_conf,
            reasoning_ready=reasoning_ready,
            missing_evidence=["video"] if next_action == "invoke_video" else [],
            recommendations=[f"Recommended action: {next_action}"],
            explanations=explanations
        )
        
        return ConfidenceResult(
            report=report,
            next_action=next_action,
            requires_clarification=needs_clarification,
            ready_for_response=(next_action == "respond")
        )

    def _evaluate_sources(self, bundle: EvidenceBundle, explanations: list[ConfidenceExplanation]):
        meta_scores = [e.confidence for e in bundle.evidence if e.source == "postgres_metadata"]
        vec_scores = [e.confidence for e in bundle.evidence if e.source == "milvus_vector"]
        vid_scores = [e.confidence for e in bundle.evidence if e.source == "s3_video"]
        
        meta_conf = sum(meta_scores) / len(meta_scores) if meta_scores else None
        vec_conf = sum(vec_scores) / len(vec_scores) if vec_scores else None
        vid_conf = sum(vid_scores) / len(vid_scores) if vid_scores else None
        
        if meta_conf is not None:
            explanations.append(ConfidenceExplanation(factor="source_metadata", score=meta_conf, explanation="Metadata retrieved from strict DB schema."))
        if vec_conf is not None:
            explanations.append(ConfidenceExplanation(factor="source_vector", score=vec_conf, explanation=f"Vector similarity averages {vec_conf:.2f}."))
            
        return meta_conf, vec_conf, vid_conf

    def _evaluate_agreement(self, bundle: EvidenceBundle, explanations: list[ConfidenceExplanation]) -> float:
        if not bundle.evidence:
            return 0.0
            
        # Simplified agreement logic: if multiple sources reference the same camera in a short time frame
        cameras = set()
        for e in bundle.evidence:
            cam_id = e.metadata.get("camera_id")
            if cam_id:
                cameras.add(cam_id)
                
        # If we have multiple evidence items pointing to the same camera, agreement is high
        if len(bundle.evidence) > 1 and len(cameras) < len(bundle.evidence):
            score = 0.95
            explanations.append(ConfidenceExplanation(factor="agreement", score=score, explanation="Multiple sources agree on the same camera/event location."))
            return score
            
        score = 0.5
        explanations.append(ConfidenceExplanation(factor="agreement", score=score, explanation="Little overlap between evidence sources."))
        return score

    def _evaluate_temporal(self, bundle: EvidenceBundle, explanations: list[ConfidenceExplanation]) -> float:
        if not bundle.evidence:
            return 0.0
            
        # Check if timeline is ordered properly (EvidenceBundle already sorts it, so it's always consistent)
        score = 1.0
        explanations.append(ConfidenceExplanation(factor="temporal", score=score, explanation="Timeline is chronologically consistent."))
        return score

    def _evaluate_completeness(self, bundle: EvidenceBundle, intent: str, explanations: list[ConfidenceExplanation]) -> float:
        if not bundle.evidence:
            score = 0.0
            explanations.append(ConfidenceExplanation(factor="completeness", score=score, explanation="No evidence found."))
            return score
            
        sources = {e.source for e in bundle.evidence}
        
        # For person search, we want vector matches
        if intent == Intent.PERSON_SEARCH.value:
            if "milvus_vector" in sources:
                score = 0.9
                explanations.append(ConfidenceExplanation(factor="completeness", score=score, explanation="Vector matches found for person search."))
            else:
                score = 0.3
                explanations.append(ConfidenceExplanation(factor="completeness", score=score, explanation="Missing vector matches for person search."))
            return score
            
        score = 0.8 # Default
        explanations.append(ConfidenceExplanation(factor="completeness", score=score, explanation="Sufficient evidence gathered."))
        return score

    def _determine_next_action(self, overall_score: float, completeness_score: float, intent: str):
        if overall_score < self.policy.reject:
            return False, "reject_query", True
            
        if overall_score < self.policy.clarification:
            return False, "ask_clarification", True
            
        if intent in [Intent.EVENT_SEARCH.value, Intent.PERSON_SEARCH.value] and completeness_score < 1.0:
            # Need visual confirmation
            return True, "invoke_video", False
            
        return True, "respond", False
