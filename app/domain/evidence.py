import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

class BaseEvidence(BaseModel):
    evidence_id: UUID = Field(default_factory=uuid4)
    evidence_type: "EvidenceType | None" = None
    source: str
    source_id: str | None = None
    confidence: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp: datetime
    trace_id: UUID | None = None
    checksum: str | None = None
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    provenance_sources: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, str]] = Field(default_factory=list) # e.g. [{"type": "appears_in", "target_id": "vid_1"}]
    
    def generate_hash(self) -> str:
        # A simple hash implementation for deduplication
        # Inheriting classes can override if they need specific fields
        content = f"{self.source}_{self.timestamp.isoformat()}_{self.metadata.get('camera_id', '')}"
        return hashlib.sha256(content.encode()).hexdigest()

    def generate_semantic_hash(self) -> str:
        # A hash implementation for semantic deduplication regardless of source
        cam_id = str(self.metadata.get('camera_id', '')).strip().lower()
        
        # Normalize timestamp (truncate to second)
        norm_ts = self.timestamp.replace(microsecond=0).isoformat()
        
        # Normalize description
        desc = str(self.metadata.get('description', '')).strip().lower()
        import re
        desc = re.sub(r'\s+', ' ', desc)
        
        # Include spatial/track ID if available
        track_id = str(self.metadata.get('track_id', '')).strip().lower()
        bbox = str(self.metadata.get('bbox', ''))
        
        content = f"{cam_id}_{norm_ts}_{desc}_{track_id}_{bbox}"
        return hashlib.sha256(content.encode()).hexdigest()

class MetadataEvidence(BaseEvidence):
    pass

class PersonEvidence(BaseEvidence):
    pass

class VehicleEvidence(BaseEvidence):
    pass

class VideoEvidence(BaseEvidence):
    pass

class EvidenceBundle(BaseModel):
    evidence: list[BaseEvidence] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    
    def add_evidence(self, new_evidence: BaseEvidence):
        # Deduplication check by evidence_id
        for e in self.evidence:
            if str(e.evidence_id) == str(new_evidence.evidence_id):
                return # Duplicate found, skip adding
                
        # Deduplication check by semantic hash
        new_hash = new_evidence.generate_semantic_hash()
        for e in self.evidence:
            if e.generate_semantic_hash() == new_hash:
                # Same observation from potentially a different source. Merge provenance.
                new_prov_source = {
                    "source": new_evidence.source,
                    "evidence_id": str(new_evidence.evidence_id),
                    "confidence": new_evidence.confidence,
                    "provenance": new_evidence.provenance
                }
                
                # Make sure we don't add the exact same source twice
                if not e.provenance_sources:
                    # Initialize with original source
                    e.provenance_sources.append({
                        "source": e.source,
                        "evidence_id": str(e.evidence_id),
                        "confidence": e.confidence,
                        "provenance": e.provenance
                    })
                    
                source_exists = any(ps.get("source") == new_evidence.source for ps in e.provenance_sources)
                if not source_exists:
                    e.provenance_sources.append(new_prov_source)
                return # Semantic duplicate merged, skip adding as new evidence
        
        self.evidence.append(new_evidence)
        self._sort_chronologically()
        
    def _sort_chronologically(self):
        self.evidence.sort(key=lambda x: x.timestamp)
        
    def get_timeline(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "source": e.source,
                "confidence": e.confidence,
                "summary": e.metadata.get("description", "Event occurred")
            }
            for e in self.evidence
        ]

from app.domain.models.enums import EvidenceType
BaseEvidence.model_rebuild()
MetadataEvidence.model_rebuild()
PersonEvidence.model_rebuild()
VehicleEvidence.model_rebuild()
VideoEvidence.model_rebuild()
