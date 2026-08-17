from abc import ABC, abstractmethod
from typing import List
from app.schemas.evidence_contract import EvidenceContract

class EvidenceOutputInterface(ABC):
    """
    Interface for outputting EvidenceContracts from the CV Pipeline.
    This establishes the boundary between CV and VISTA Agentic RAG.
    """
    
    @abstractmethod
    def push_evidence(self, evidence: List[EvidenceContract]):
        """Push a batch of EvidenceContract observations downstream."""
        pass
