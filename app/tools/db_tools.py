from typing import Any, Optional
from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool
from app.services.db_services import EvidenceService, TrackService

class EvidenceSearchTool(BaseTool):
    def __init__(self, evidence_service: EvidenceService):
        self._name = "evidence_search_tool"
        self._description = "Searches for canonical evidence in PostgreSQL."
        self.service = evidence_service

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, context: VistaContext, **kwargs) -> Any:
        video_id = kwargs.get("video_id")
        camera_id = kwargs.get("camera_id")
        track_id = kwargs.get("track_id")
        
        if not video_id:
            raise ValueError("video_id is required for EvidenceSearchTool")
            
        return await self.service.search_evidence(video_id, camera_id, track_id)

    def validate(self, **kwargs) -> bool:
        return "video_id" in kwargs

    async def health(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        return {"category": "database", "entity": "evidence"}


class TrackSearchTool(BaseTool):
    def __init__(self, track_service: TrackService):
        self._name = "track_search_tool"
        self._description = "Fetches a track summary from PostgreSQL."
        self.service = track_service

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, context: VistaContext, **kwargs) -> Any:
        video_id = kwargs.get("video_id")
        track_id = kwargs.get("track_id")
        
        if not video_id or not track_id:
            raise ValueError("video_id and track_id are required")
            
        return await self.service.get_track_summary(video_id, track_id)

    def validate(self, **kwargs) -> bool:
        return "video_id" in kwargs and "track_id" in kwargs

    async def health(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        return {"category": "database", "entity": "track"}


class TimelineTool(BaseTool):
    def __init__(self, evidence_service: EvidenceService):
        self._name = "timeline_tool"
        self._description = "Fetches temporally ordered evidence for timeline construction."
        self.service = evidence_service

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, context: VistaContext, **kwargs) -> Any:
        video_id = kwargs.get("video_id")
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        
        # Will expand in future phases to do robust temporal bounding box lookups in MongoDB
        # For Phase 2, we fetch the base evidence
        return await self.service.search_evidence(
            video_id=video_id,
            start_time_sec=start_time,
            end_time_sec=end_time
        )

    def validate(self, **kwargs) -> bool:
        return "video_id" in kwargs

    async def health(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        return {"category": "database", "entity": "timeline"}
