from abc import ABC, abstractmethod
from typing import Any, List, Optional
from uuid import UUID

from app.schemas.evidence_contract import EvidenceContract


class EvidenceRepository(ABC):
    """
    Authoritative repository for Evidence metadata.
    Typically implemented over PostgreSQL.
    """
    @abstractmethod
    async def create(self, evidence: EvidenceContract) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, evidence_id: str | UUID) -> Optional[EvidenceContract]:
        pass

    @abstractmethod
    async def search(
        self, 
        video_id: str, 
        camera_id: Optional[str] = None, 
        track_id: Optional[str] = None,
        start_time_sec: Optional[float] = None,
        end_time_sec: Optional[float] = None
    ) -> List[EvidenceContract]:
        pass


class ObservationRepository(ABC):
    """
    High-volume repository for raw tracking observations.
    Typically implemented over MongoDB.
    """
    @abstractmethod
    async def insert_observation(self, observation: dict[str, Any]) -> None:
        """
        observation must contain: evidence_id, video_id, camera_id, track_id, 
        frame_index, timestamp_sec, bbox, confidence, crop_uri.
        """
        pass
        
    @abstractmethod
    async def get_observations_for_evidence(self, evidence_id: str | UUID) -> List[dict[str, Any]]:
        pass


class PersonRepository(ABC):
    @abstractmethod
    async def create_person(self) -> str:
        """Atomically create a new canonical person and return the generated ID."""
        pass


class TrackRepository(ABC):
    """
    Repository for managing tracks globally.
    Typically implemented over PostgreSQL.
    """
    @abstractmethod
    async def get_track_summary(self, video_id: str, track_id: str) -> Optional[dict[str, Any]]:
        pass
        
    @abstractmethod
    async def assign_person_to_track(self, video_id: str, track_id: str, person_id: str) -> None:
        """Assigns a resolved canonical_person_id to a specific track."""
        pass


class CameraRepository(ABC):
    """
    Repository for managing Camera configurations.
    Typically implemented over PostgreSQL.
    """
    @abstractmethod
    async def get_camera(self, camera_id: str) -> Optional[dict[str, Any]]:
        pass


class VideoRepository(ABC):
    """
    Repository for managing Video Segment metadata.
    Typically implemented over PostgreSQL.
    """
    @abstractmethod
    async def get_video(self, video_id: str) -> Optional[dict[str, Any]]:
        pass
