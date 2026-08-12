import hashlib
from typing import Any


class VideoCache:
    """
    Caches expensive VLM reasoning results.
    Key is based on camera, timestamp, duration, and prompt.
    """
    def __init__(self):
        self._cache = {} # Mock Redis
        
    def _generate_key(self, camera_id: str, start_time: str, duration: int, prompt: str) -> str:
        content = f"{camera_id}_{start_time}_{duration}_{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, camera_id: str, start_time: str, duration: int, prompt: str) -> dict[str, Any] | None:
        key = self._generate_key(camera_id, start_time, duration, prompt)
        return self._cache.get(key)

    def set(self, camera_id: str, start_time: str, duration: int, prompt: str, result: dict[str, Any]):
        key = self._generate_key(camera_id, start_time, duration, prompt)
        self._cache[key] = result
