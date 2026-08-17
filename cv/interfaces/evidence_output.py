"""
CV Output Interface — defines the contract between Computer Vision and Agentic RAG.

Phase 0: Interface definition only.
Phase 2: CV pipeline implements this interface.

The RAG layer consumes EvidenceContract objects.
It does not care whether they were produced by:
  - YOLO + ByteTrack + OSNet
  - RT-DETR + BoT-SORT + another ReID model
  - Manual annotations
  - Synthetic test data

That abstraction is the entire point of this interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.schemas.evidence_contract import EvidenceContract


@runtime_checkable
class CVOutputInterface(Protocol):
    """
    Interface that any CV pipeline must implement to feed the Agentic RAG.

    Phase 2 will implement:
        Video → Detection → Tracking → ReID → Attributes → Events → EvidenceContract

    Usage:
        class MyCVPipeline:
            def produce_evidence(self, video_path: str) -> list[EvidenceContract]:
                # Run detection, tracking, ReID, attribute extraction, event detection
                # Return list of EvidenceContract objects
                ...
    """

    def produce_evidence(self, video_path: str) -> list[EvidenceContract]:
        """
        Process a video file and produce a list of evidence contracts.

        Args:
            video_path: Path to the video file to process.

        Returns:
            List of EvidenceContract objects, one per observation.
            Each observation has:
              - evidence_id: unique observation identifier
              - provenance: where this evidence came from
              - subject: what entity this describes (with optional track_id)
              - attributes: structured attributes (gender, age_group, clothing, etc.)
              - observation: free-form observation data
              - confidence: model confidence score
        """
        ...
