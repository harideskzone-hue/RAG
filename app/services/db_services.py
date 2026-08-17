from typing import Any, List, Optional
from uuid import UUID

from app.domain.repositories.base import (
    EvidenceRepository, 
    ObservationRepository, 
    TrackRepository, 
    VideoRepository
)
from app.domain.storage.blob import BlobStorage
from app.schemas.evidence_contract import EvidenceContract


class StorageService:
    def __init__(self, blob_storage: BlobStorage):
        self.blob = blob_storage

    async def store_crop(self, video_id: str, track_id: str, crop_id: str, image_bytes: bytes) -> str:
        return await self.blob.save_crop(video_id, track_id, crop_id, image_bytes)

    async def get_crop(self, uri: str) -> Optional[bytes]:
        return await self.blob.get_crop(uri)


class EvidenceService:
    def __init__(self, evidence_repo: EvidenceRepository, obs_repo: ObservationRepository):
        self.evidence_repo = evidence_repo
        self.obs_repo = obs_repo

    async def save_evidence(self, evidence: EvidenceContract, observation_data: dict[str, Any]) -> None:
        """
        Idempotent save: 
        1. Save metadata to postgres (evidence_repo).
        2. Save raw high-volume observation to mongodb (obs_repo).
        """
        # Save to PG (source of truth)
        # Postgres repository handles idempotency via unique UUID or ON CONFLICT clauses
        await self.evidence_repo.create(evidence)
        
        # Save to Mongo (observation store)
        await self.obs_repo.insert_observation(observation_data)

    async def search_evidence(
        self, 
        video_id: str, 
        camera_id: Optional[str] = None, 
        track_id: Optional[str] = None,
        start_time_sec: Optional[float] = None,
        end_time_sec: Optional[float] = None
    ) -> List[EvidenceContract]:
        return await self.evidence_repo.search(video_id, camera_id, track_id, start_time_sec, end_time_sec)


class TrackService:
    def __init__(self, track_repo: TrackRepository):
        self.track_repo = track_repo

    async def get_track_summary(self, video_id: str, track_id: str) -> Optional[dict[str, Any]]:
        return await self.track_repo.get_track_summary(video_id, track_id)


class VideoService:
    def __init__(self, video_repo: VideoRepository):
        self.video_repo = video_repo

    async def get_video_info(self, video_id: str) -> Optional[dict[str, Any]]:
        return await self.video_repo.get_video(video_id)
