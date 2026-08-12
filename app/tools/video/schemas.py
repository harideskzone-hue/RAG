from typing import Any

from pydantic import BaseModel


class VideoToolResult(BaseModel):
    success: bool = True
    video_uri: str = ""
    metadata: dict[str, Any] = {}
    error: str = ""
