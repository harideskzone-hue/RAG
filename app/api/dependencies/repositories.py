from fastapi import Depends

from app.graph.supervisor.event_bus import EventBus
from app.services.repositories.alert_repository import AlertRepository
from app.services.repositories.camera_repository import CameraRepository
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository
from app.tools.metadata.postgres_tool import PostgresTool
from app.tools.vector.vector_tool import VectorTool
from app.tools.vector.milvus_tool import MilvusTool

# Application-scoped singleton — shared across all requests
_event_bus = EventBus()


def _is_depends(obj) -> bool:
    return obj is None or hasattr(obj, "dependency") or "params.Depends" in str(type(obj))

def get_event_bus() -> EventBus:
    return _event_bus

def get_postgres_tool(event_bus: EventBus = Depends(get_event_bus)) -> PostgresTool:
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    return PostgresTool(event_bus)

def get_vector_tool(event_bus: EventBus = Depends(get_event_bus)) -> VectorTool:
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    return VectorTool(event_bus)

# Backwards compatibility alias
get_milvus_tool = get_vector_tool

def get_camera_repository(postgres_tool: PostgresTool = Depends(get_postgres_tool)) -> CameraRepository:
    if _is_depends(postgres_tool):
        postgres_tool = get_postgres_tool()
    return CameraRepository(postgres_tool)

def get_alert_repository(postgres_tool: PostgresTool = Depends(get_postgres_tool)) -> AlertRepository:
    if _is_depends(postgres_tool):
        postgres_tool = get_postgres_tool()
    return AlertRepository(postgres_tool)

def get_person_repository(vector_tool: VectorTool = Depends(get_vector_tool)) -> PersonRepository:
    if _is_depends(vector_tool):
        vector_tool = get_vector_tool()
    return PersonRepository(vector_tool)

def get_vehicle_repository(vector_tool: VectorTool = Depends(get_vector_tool)) -> VehicleRepository:
    if _is_depends(vector_tool):
        vector_tool = get_vector_tool()
    return VehicleRepository(vector_tool)
