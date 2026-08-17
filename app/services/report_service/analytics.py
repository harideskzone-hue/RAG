from typing import Any

from app.domain.evidence import EvidenceBundle


class AnalyticsEngine:
    """
    Calculates KPIs from the EvidenceBundle.
    """
    def calculate(self, bundle: EvidenceBundle) -> dict[str, Any]:
        return {
            "total_incidents": len(bundle.evidence),
            "critical_incidents": len([e for e in bundle.evidence if (getattr(e.confidence, 'overall', e.confidence) if hasattr(e.confidence, 'overall') else e.confidence) > 0.9])
        }

class StatisticsEngine:
    """
    Generates counts and trends.
    """
    def generate(self, bundle: EvidenceBundle) -> dict[str, Any]:
        sources = {}
        for e in bundle.evidence:
            sources[e.source] = sources.get(e.source, 0) + 1
        return {"sources": sources}

class NarrativeEngine:
    """
    Generates natural language summaries.
    """
    def generate(self, bundle: EvidenceBundle, analytics: dict, stats: dict) -> str:
        return f"Report summary: {analytics['total_incidents']} incidents recorded across {len(stats['sources'])} sources."
