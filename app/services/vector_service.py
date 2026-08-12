import time
from enum import Enum

from app.domain.models import PersonMatch, VehicleMatch
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.schemas.context import VistaContext
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository


class RetrievalMode(Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    HIGH_PRECISION = "HIGH_PRECISION"

class VectorService:
    """
    Composes Repositories, handles retrieval strategies, and emits service-level telemetry.
    The Agent calls this service.
    """
    def __init__(self, person_repo: PersonRepository, vehicle_repo: VehicleRepository, event_bus: EventBus):
        self.person_repo = person_repo
        self.vehicle_repo = vehicle_repo
        self.event_bus = event_bus

    def _get_top_k_for_mode(self, mode: RetrievalMode) -> int:
        if mode == RetrievalMode.FAST:
            return 5
        elif mode == RetrievalMode.HIGH_PRECISION:
            return 50 # Assume re-ranking logic would follow, or just deeper search
        return 15 # BALANCED

    async def search_person(self, embedding: list[float], mode: RetrievalMode, context: VistaContext) -> list[PersonMatch]:
        start_time = time.time()
        self._publish_event("SERVICE_START", "vector_service", context.conversation_id, start_time)
        
        top_k = self._get_top_k_for_mode(mode)
        
        try:
            matches = await self.person_repo.search_person(embedding, top_k, context)
            self._publish_event("SERVICE_COMPLETE", "vector_service", context.conversation_id, start_time, end_time=time.time())
            return matches
        except Exception as e:
            self._publish_event("SERVICE_ERROR", "vector_service", context.conversation_id, start_time, end_time=time.time(), error=str(e))
            raise

    async def search_vehicle(self, embedding: list[float], mode: RetrievalMode, context: VistaContext) -> list[VehicleMatch]:
        start_time = time.time()
        self._publish_event("SERVICE_START", "vector_service", context.conversation_id, start_time)
        
        top_k = self._get_top_k_for_mode(mode)
        
        try:
            matches = await self.vehicle_repo.search_vehicle(embedding, top_k, context)
            self._publish_event("SERVICE_COMPLETE", "vector_service", context.conversation_id, start_time, end_time=time.time())
            return matches
        except Exception as e:
            self._publish_event("SERVICE_ERROR", "vector_service", context.conversation_id, start_time, end_time=time.time(), error=str(e))
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
