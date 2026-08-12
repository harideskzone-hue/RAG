from fastapi import Depends

from app.graph.supervisor.event_bus import EventBus
from app.services.repositories.alert_repository import AlertRepository
from app.services.repositories.camera_repository import CameraRepository
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository
from app.tools.metadata.postgres_tool import PostgresTool
from app.tools.vector.milvus_tool import MilvusTool

# Application-scoped singleton — shared across all requests
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    return _event_bus

def get_postgres_tool(event_bus: EventBus = Depends(get_event_bus)) -> PostgresTool:
    return PostgresTool(event_bus)

def get_milvus_tool(event_bus: EventBus = Depends(get_event_bus)) -> MilvusTool:
    return MilvusTool(event_bus)

def get_camera_repository(postgres_tool: PostgresTool = Depends(get_postgres_tool)) -> CameraRepository:
    return CameraRepository(postgres_tool)

def get_alert_repository(postgres_tool: PostgresTool = Depends(get_postgres_tool)) -> AlertRepository:
    return AlertRepository(postgres_tool)

def get_person_repository(milvus_tool: MilvusTool = Depends(get_milvus_tool)) -> PersonRepository:
    return PersonRepository(milvus_tool)

def get_vehicle_repository(milvus_tool: MilvusTool = Depends(get_milvus_tool)) -> VehicleRepository:
    return VehicleRepository(milvus_tool)
