from app.domain.evidence import EvidenceBundle
from app.schemas.context import BaseResult


class EvidenceResult(BaseResult):
    """
    Result wrapper for Evidence Collector.
    """
    bundle: EvidenceBundle
