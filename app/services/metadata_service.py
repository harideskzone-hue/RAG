import time

from app.domain.models import Alert, Camera
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.schemas.context import VistaContext
from app.services.repositories.alert_repository import AlertRepository
from app.services.repositories.camera_repository import CameraRepository


class MetadataService:
    """
    Composes Repositories, adds caching, and emits service-level telemetry.
    The Agent calls this service.
    """
    def __init__(self, camera_repo: CameraRepository, alert_repo: AlertRepository, event_bus: EventBus):
        self.camera_repo = camera_repo
        self.alert_repo = alert_repo
        self.event_bus = event_bus
        self._cache = {} # Simple in-memory cache fallback

    async def get_camera_status(self, camera_id: str, context: VistaContext) -> Camera | None:
        start_time = time.time()
        self._publish_event("SERVICE_START", "metadata_service", context.conversation_id, start_time)
        
        # Check cache
        cache_key = f"camera_{camera_id}"
        if cache_key in self._cache:
            self._publish_event("CACHE_HIT", "metadata_service", context.conversation_id, start_time)
            return self._cache[cache_key]
            
        self._publish_event("CACHE_MISS", "metadata_service", context.conversation_id, start_time)
        
        # Fetch from repository
        try:
            camera = await self.camera_repo.get_camera(camera_id, context)
            if camera:
                self._cache[cache_key] = camera # Cache it
            self._publish_event("SERVICE_COMPLETE", "metadata_service", context.conversation_id, start_time, end_time=time.time())
            return camera
        except Exception as e:
            self._publish_event("SERVICE_ERROR", "metadata_service", context.conversation_id, start_time, end_time=time.time(), error=str(e))
            raise

    async def get_all_cameras(self, context: VistaContext) -> list[Camera]:
        start_time = time.time()
        self._publish_event("SERVICE_START", "metadata_service", context.conversation_id, start_time)
        
        cache_key = "all_cameras"
        if cache_key in self._cache:
            self._publish_event("CACHE_HIT", "metadata_service", context.conversation_id, start_time)
            return self._cache[cache_key]
            
        self._publish_event("CACHE_MISS", "metadata_service", context.conversation_id, start_time)
        
        try:
            cameras = await self.camera_repo.get_all_cameras(context)
            self._cache[cache_key] = cameras
            self._publish_event("SERVICE_COMPLETE", "metadata_service", context.conversation_id, start_time, end_time=time.time())
            return cameras
        except Exception as e:
            self._publish_event("SERVICE_ERROR", "metadata_service", context.conversation_id, start_time, end_time=time.time(), error=str(e))
            raise

    async def get_recent_alerts(self, limit: int, context: VistaContext) -> list[Alert]:
        start_time = time.time()
        self._publish_event("SERVICE_START", "metadata_service", context.conversation_id, start_time)
        try:
            alerts = await self.alert_repo.get_recent_alerts(limit, context)
            self._publish_event("SERVICE_COMPLETE", "metadata_service", context.conversation_id, start_time, end_time=time.time())
            return alerts
        except Exception as e:
            self._publish_event("SERVICE_ERROR", "metadata_service", context.conversation_id, start_time, end_time=time.time(), error=str(e))
            raise

    def _publish_event(self, event_type: str, agent_name: str, trace_id: str, start_time: float, end_time: float = None, error: str = None):
        kwargs = {
            "agent_name": agent_name,
            "event_type": event_type,
            "start_time": start_time,
            "status": "ERROR" if error else "SUCCESS",
            "trace_id": trace_id,
        }
        if end_time:
            kwargs["end_time"] = end_time
            kwargs["latency_ms"] = (end_time - start_time) * 1000
        if error:
            kwargs["errors"] = [error]
            
        self.event_bus.publish(AgentEvent(**kwargs))
