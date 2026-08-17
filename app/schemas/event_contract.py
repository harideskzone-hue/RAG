from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class IncidentEventType(str, Enum):
    CHAIN_SNATCHING_ROBBERY = "CHAIN_SNATCHING_ROBBERY"
    ROBBERY = "ROBBERY"
    THEFT = "THEFT"
    FIGHT = "FIGHT"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    LOITERING = "LOITERING"
    UNAUTHORIZED_ENTRY = "UNAUTHORIZED_ENTRY"
    ACCIDENT = "ACCIDENT"
    CROWD = "CROWD"
    FIRE = "FIRE"
    ABSTAIN = "ABSTAIN"
    UNKNOWN = "UNKNOWN"


class DetectedEvent(BaseModel):
    """
    Proposed semantic interpretation from Qwen LLM.
    """
    event_type: IncidentEventType = Field(..., description="Semantic incident classification")
    confidence: float = Field(..., description="Confidence score [0.0, 1.0]")
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")
    track_ids: List[str] = Field(default_factory=list, description="Associated tracklet IDs")
    reason: str = Field(..., description="Natural language explanation of observed behavior")
    severity: str = Field(default="MEDIUM", description="Severity level: LOW, MEDIUM, HIGH, CRITICAL")


class VerifiedEventContract(BaseModel):
    """
    Authoritative, deterministic event contract verified against CV evidence.
    """
    event_id: str
    event_type: str
    camera_id: str
    video_id: str
    start_time: float
    end_time: float
    duration_sec: float
    track_ids: List[str]
    canonical_person_ids: List[str]
    confidence: float
    severity: str
    clip_path: str
    clip_url: str
    thumbnail_path: str
    thumbnail_url: str
    reason: str
    clip_sha256: str
    provenance: Dict[str, Any]
