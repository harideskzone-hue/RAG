import os
import aiofiles
from typing import Optional
from pathlib import Path

from app.config.db import db_settings
from app.domain.storage.blob import BlobStorage

class LocalBlobStorage(BlobStorage):
    """
    Local filesystem implementation of BlobStorage.
    Designed for Phase 2 before transitioning to MinIO/S3.
    """
    def __init__(self):
        self.base_dir = Path(db_settings.STORAGE_BASE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_crop(self, video_id: str, track_id: str, crop_id: str, image_bytes: bytes) -> str:
        """
        Pattern: tracks/{video_id}/{track_id}/crops/{crop_id}.jpg
        """
        rel_path = f"tracks/{video_id}/{track_id}/crops/{crop_id}.jpg"
        full_path = self.base_dir / rel_path
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, 'wb') as f:
            await f.write(image_bytes)
            
        return rel_path
        
    async def copy_crop_to_person(self, original_uri: str, person_id: str, crop_id: str) -> str:
        import shutil
        original_path = self.base_dir / original_uri
        rel_path = f"persons/{person_id}/crops/{crop_id}.jpg"
        new_path = self.base_dir / rel_path
        
        if original_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy instead of move, preserving the original evidence
            shutil.copy2(original_path, new_path)
            
        return rel_path

    async def get_crop(self, uri: str) -> Optional[bytes]:
        full_path = self.base_dir / uri
        if not full_path.exists():
            return None
            
        async with aiofiles.open(full_path, 'rb') as f:
            return await f.read()

    async def save_segment(self, video_id: str, segment_id: str, video_bytes: bytes) -> str:
        """
        Pattern: videos/{video_id}/segments/{segment_id}.mp4
        """
        rel_path = f"videos/{video_id}/segments/{segment_id}.mp4"
        full_path = self.base_dir / rel_path
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, 'wb') as f:
            await f.write(video_bytes)
            
        return rel_path

    async def get_segment(self, uri: str) -> Optional[bytes]:
        full_path = self.base_dir / uri
        if not full_path.exists():
            return None
            
        async with aiofiles.open(full_path, 'rb') as f:
            return await f.read()

local_storage = LocalBlobStorage()
