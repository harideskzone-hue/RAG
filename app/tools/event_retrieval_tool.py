import logging
from typing import List, Optional, Dict, Any

from app.schemas.event_contract import VerifiedEventContract
from app.services.repositories.event_repository import EventRepository

logger = logging.getLogger("EventRetrievalTool")


class EventRetrievalTool:
    """
    Deterministic Agentic RAG tool for retrieving verified incident events and evidence video clips.
    """

    def __init__(self):
        self.repository = EventRepository()

    async def search_events(
        self,
        event_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        video_id: Optional[str] = None,
        canonical_person_id: Optional[str] = None,
        limit: int = 5,
        db_session = None
    ) -> List[VerifiedEventContract]:
        """
        Queries multi-store database and local verified storage for matching incident events.
        """
        return await self.repository.get_events(
            event_type=event_type,
            camera_id=camera_id,
            video_id=video_id,
            canonical_person_id=canonical_person_id,
            limit=limit,
            db_session=db_session
        )
