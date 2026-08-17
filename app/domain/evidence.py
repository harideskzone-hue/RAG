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
    timestamp: datetime | str
    trace_id: UUID | None = None
    checksum: str | None = None
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    provenance_sources: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, str]] = Field(default_factory=list) # e.g. [{"type": "appears_in", "target_id": "vid_1"}]
    
    def generate_hash(self) -> str:
        ts_str = self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp)
        content = f"{self.source}_{ts_str}_{self.metadata.get('camera_id', '')}"
        return hashlib.sha256(content.encode()).hexdigest()

    def generate_semantic_hash(self) -> str:
        cam_id = str(self.metadata.get('camera_id', '')).strip().lower()
        if isinstance(self.timestamp, datetime):
            norm_ts = self.timestamp.replace(microsecond=0).isoformat()
        else:
            norm_ts = str(self.timestamp)
        
        desc = str(self.metadata.get('description', '')).strip().lower()
        import re
        desc = re.sub(r'\s+', ' ', desc)
        
        track_id = str(self.metadata.get('track_id', '')).strip().lower()
        bbox = str(self.metadata.get('bbox', ''))
        
        content = f"{cam_id}_{norm_ts}_{desc}_{track_id}_{bbox}"
        return hashlib.sha256(content.encode()).hexdigest()

    def to_contract(self) -> "EvidenceContract":
        """
        Convert this domain evidence into a canonical EvidenceContract.
        
        Maps the metadata dict structure into the typed contract schema.
        Provenance validation happens at the schema level — fabricated
        values will raise ValueError.
        """
        from app.schemas.evidence_contract import (
            EvidenceContract,
            EvidenceProvenance,
            EvidenceSubject,
            EvidenceAttributes,
        )

        meta = self.metadata if isinstance(self.metadata, dict) else {}
        origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}
        attrs = meta.get("attributes") if isinstance(meta.get("attributes"), dict) else {}

        provenance = EvidenceProvenance(
            video_id=origin.get("video_id") or meta.get("video_id"),
            camera_id=origin.get("camera_id") or meta.get("camera_id"),
            track_id=origin.get("track_id") or meta.get("track_id"),
            source_type=origin.get("type", self.source),
            video_timestamp_sec=origin.get("video_timestamp_sec"),
            frame_number=origin.get("frame_number"),
        )

        subject = EvidenceSubject(
            entity_type=str(getattr(self, "evidence_type", "") or ""),
            track_id=origin.get("track_id") or meta.get("track_id"),
            description=meta.get("description", ""),
        )

        attributes = EvidenceAttributes(
            gender=attrs.get("gender"),
            age_group=attrs.get("age_group"),
            clothing_upper=attrs.get("clothing_upper"),
            clothing_lower=attrs.get("clothing_lower"),
            clothing_color=attrs.get("clothing_color"),
            hair_style=attrs.get("hair_style"),
            hair_color=attrs.get("hair_color"),
            vehicle_type=attrs.get("vehicle_type"),
            vehicle_color=attrs.get("vehicle_color"),
            license_plate=attrs.get("license_plate"),
            behavior=attrs.get("behavior"),
            location=attrs.get("location"),
        )

        return EvidenceContract(
            evidence_id=self.evidence_id,
            provenance=provenance,
            subject=subject,
            attributes=attributes,
            observation=meta,
            confidence=self.confidence,
            source=self.source,
        )

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
        def _tz_aware_key(e):
            ts = e.timestamp
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts.timestamp()
            if isinstance(ts, (int, float)):
                return float(ts)
            if isinstance(ts, str):
                import re
                match = re.search(r'(\d+(?:\.\d+)?)s', ts)
                if match:
                    return float(match.group(1))
                try:
                    return float(ts)
                except Exception:
                    pass
            return 0.0
        self.evidence.sort(key=_tz_aware_key)
        
    def get_timeline(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp.isoformat() if isinstance(e.timestamp, datetime) else str(e.timestamp),
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
