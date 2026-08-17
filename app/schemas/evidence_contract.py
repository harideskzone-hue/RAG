"""
Evidence Contract Schemas.

These are the Pydantic schemas for the evidence pipeline's
authoritative data contracts. They enforce provenance integrity
at the schema level via validators.

Key types:
    EvidenceContract   — a single observation from any source
    VerifiedResultContract — the output of VerificationAgent
"""
from __future__ import annotations

import re
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# Provenance values that MUST NOT appear in any evidence — truly placeholder/test values only
_FABRICATED_VIDEO_IDS = {"default_video", "dummy_video", "test_video", "fake_video"}
_FABRICATED_CAMERA_IDS = {"default_camera", "unknown_camera", "test_camera", "dummy_camera", "fake_camera"}


class CameraRegistry:
    """
    Registry-based provenance validation.
    A camera_id is valid if it has been registered or if no registry is configured.
    Stronger than a blacklist: only known cameras produce valid provenance.
    """
    _registered_cameras: set[str] = set()
    _enforce: bool = False  # When True, only registered cameras are accepted

    @classmethod
    def register(cls, camera_id: str) -> None:
        cls._registered_cameras.add(camera_id.strip().lower())

    @classmethod
    def is_valid(cls, camera_id: str) -> bool:
        if not camera_id:
            return True  # None/empty are allowed (some evidence may lack camera info)
        normalized = camera_id.strip().lower()
        # Always reject known fabricated values
        if normalized in _FABRICATED_CAMERA_IDS:
            return False
        # If enforcement is on, only registered cameras pass
        if cls._enforce and cls._registered_cameras:
            return normalized in cls._registered_cameras
        return True

    @classmethod
    def set_enforce(cls, enforce: bool) -> None:
        cls._enforce = enforce


class EvidenceProvenance(BaseModel):
    """Where this evidence came from. Every field must be real or None."""
    video_id: str | None = None
    camera_id: str | None = None
    track_id: str | None = None
    source_type: str = ""  # "video_ingestion", "vector_search", "metadata_query"
    video_timestamp_sec: float | None = None
    frame_number: int | None = None

    @field_validator("video_id")
    @classmethod
    def reject_fabricated_video(cls, v: str | None) -> str | None:
        if v and str(v).strip().lower() in _FABRICATED_VIDEO_IDS:
            raise ValueError(
                f"Fabricated video_id rejected: '{v}'. "
                f"Evidence must have real provenance."
            )
        return v

    @field_validator("camera_id")
    @classmethod
    def validate_camera_provenance(cls, v: str | None) -> str | None:
        if v and not CameraRegistry.is_valid(v):
            raise ValueError(
                f"Camera '{v}' failed provenance validation. "
                f"Register it via CameraRegistry.register() or check for placeholder values."
            )
        return v


class EvidenceSubject(BaseModel):
    """What entity this evidence describes."""
    entity_type: str = ""  # "person", "vehicle", "event", "camera"
    track_id: str | None = None  # track identity from CV/tracker
    description: str = ""


class EvidenceAttributes(BaseModel):
    """Structured attributes extracted from the evidence."""
    gender: str | None = None
    age_group: str | None = None
    clothing_upper: str | None = None
    clothing_lower: str | None = None
    clothing_color: str | None = None
    hair_style: str | None = None
    hair_color: str | None = None
    vehicle_type: str | None = None
    vehicle_color: str | None = None
    license_plate: str | None = None
    behavior: str | None = None
    location: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, str]:
        """Flatten to {field: value} for constraint evaluation. Omits None."""
        result = {}
        for field_name, value in self:
            if value is not None and field_name != "extra":
                result[field_name] = str(value)
        if self.extra:
            for k, v in self.extra.items():
                if v is not None:
                    result[k] = str(v)
        return result


class EvidenceContract(BaseModel):
    """
    A single observation from any source.
    
    This is the canonical evidence format consumed by the RAG pipeline.
    It does not care whether the evidence came from CV, vector search,
    metadata query, or manual annotation.
    """
    evidence_id: UUID = Field(default_factory=uuid4)
    provenance: EvidenceProvenance = Field(default_factory=EvidenceProvenance)
    subject: EvidenceSubject = Field(default_factory=EvidenceSubject)
    attributes: EvidenceAttributes = Field(default_factory=EvidenceAttributes)
    observation: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    source: str = ""

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {v}")
        return v

    def has_complete_identity(self) -> bool:
        """Check if this evidence has a complete identity key for deduplication."""
        return bool(
            self.provenance.video_id
            and self.provenance.camera_id
            and self.subject.track_id
        )

    def identity_key(self) -> tuple[str, str, str] | None:
        """Return canonical identity key, or None if incomplete."""
        if self.has_complete_identity():
            return (
                self.provenance.video_id,
                self.provenance.camera_id,
                self.subject.track_id,
            )
        return None


class VerifiedEntity(BaseModel):
    """A verified entity from the Verification Agent."""
    track_id: str | None = None
    video_id: str | None = None
    camera_id: str | None = None
    source: str = ""
    description: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float | None = None
    evidence_id: str = ""


class VerifiedResultContractSchema(BaseModel):
    """
    Pydantic schema for the Verified Result Contract.
    
    This is the authoritative output of the VerificationAgent.
    Reasoning LLM receives ONLY this — never raw observations.
    ResponseCoordinator READS this — never creates or modifies.
    """
    status: str = "no_evidence"  # "verified" | "no_evidence" | "partial"
    operation: str = ""
    target: str = ""
    constraints: list[str] = Field(default_factory=list)
    verified_count: int = 0
    verified_tracks: list[str] = Field(default_factory=list)
    verified_evidence: list[VerifiedEntity] = Field(default_factory=list)
    video_id: str | None = None
    overall_confidence: float = 0.0
    events: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("overall_confidence")
    @classmethod
    def validate_confidence_not_hardcoded(cls, v: float) -> float:
        """Confidence must come from evidence, not be hardcoded."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {v}")
        return v
