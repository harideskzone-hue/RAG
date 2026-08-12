from app.domain.evidence import EvidenceBundle


class CorrelationEngine:
    """
    Correlates evidence across time and space.
    """
    def correlate(self, bundle: EvidenceBundle) -> list[str]:
        correlations = []
        
        # Mock correlation logic
        sources = {e.source for e in bundle.evidence}
        if "postgres_metadata" in sources and "vlm_gemini" in sources:
            correlations.append("Video evidence confirms metadata alert.")
            
        return correlations
