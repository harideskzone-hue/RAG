from typing import Any

from app.domain.evidence import EvidenceBundle


class TimelineEngine:
    """
    Extracts a chronological timeline of an event.
    """
    def extract(self, bundle: EvidenceBundle) -> list[dict[str, Any]]:
        # EvidenceBundle already handles chronological sorting
        return bundle.get_timeline()
