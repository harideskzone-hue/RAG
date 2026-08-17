from abc import ABC, abstractmethod
from typing import Optional


class BlobStorage(ABC):
    """
    Storage interface for physical media (crops, video segments).
    """
    @abstractmethod
    async def save_crop(self, video_id: str, track_id: str, crop_id: str, image_bytes: bytes) -> str:
        """
        Saves a track crop image and returns its storage URI.
        Pattern: tracks/{video_id}/{track_id}/crops/{crop_id}.jpg
        """
        pass

    @abstractmethod
    async def get_crop(self, uri: str) -> Optional[bytes]:
        """
        Retrieves a crop image by its URI.
        """
        pass

    @abstractmethod
    async def save_segment(self, video_id: str, segment_id: str, video_bytes: bytes) -> str:
        """
        Saves a video segment and returns its storage URI.
        Pattern: videos/{video_id}/segments/{segment_id}.mp4
        """
        pass

    @abstractmethod
    async def get_segment(self, uri: str) -> Optional[bytes]:
        """
        Retrieves a video segment by its URI.
        """
        pass
