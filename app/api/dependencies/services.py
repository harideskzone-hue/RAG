from fastapi import Depends

from app.api.dependencies.repositories import (
    _is_depends,
    get_alert_repository,
    get_camera_repository,
    get_event_bus,
    get_person_repository,
    get_vehicle_repository,
)
from app.graph.supervisor.event_bus import EventBus
from app.services.event_service.service import EventService
from app.services.metadata_service import MetadataService
from app.services.report_service.service import ReportService
from app.services.repositories.alert_repository import AlertRepository
from app.services.repositories.camera_repository import CameraRepository
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository
from app.services.vector_service import VectorService
from app.services.video_service.service import VideoService
from app.tools.video.s3_tool import S3Tool


def get_metadata_service(
    camera_repo: CameraRepository = Depends(get_camera_repository),
    alert_repo: AlertRepository = Depends(get_alert_repository),
    event_bus: EventBus = Depends(get_event_bus)
) -> MetadataService:
    if _is_depends(camera_repo):
        camera_repo = get_camera_repository()
    if _is_depends(alert_repo):
        alert_repo = get_alert_repository()
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    return MetadataService(camera_repo, alert_repo, event_bus)

def get_vector_service(
    person_repo: PersonRepository = Depends(get_person_repository),
    vehicle_repo: VehicleRepository = Depends(get_vehicle_repository),
    event_bus: EventBus = Depends(get_event_bus)
) -> VectorService:
    if _is_depends(person_repo) or not hasattr(person_repo, "search_person"):
        person_repo = get_person_repository()
    if _is_depends(vehicle_repo) or not hasattr(vehicle_repo, "search_vehicle"):
        vehicle_repo = get_vehicle_repository()
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    return VectorService(person_repo, vehicle_repo, event_bus)

def get_s3_tool(event_bus: EventBus = Depends(get_event_bus)) -> S3Tool:
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    return S3Tool(event_bus)

def get_video_service(
    s3_tool: S3Tool = Depends(get_s3_tool),
    event_bus: EventBus = Depends(get_event_bus)
) -> VideoService:
    if _is_depends(s3_tool):
        s3_tool = get_s3_tool()
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    try:
        from app.infrastructure.llm.model_registry import ModelRegistry
        llm_client = ModelRegistry.get_client()
    except Exception:
        from app.infrastructure.llm.model_registry import ModelRegistry
        llm_client = ModelRegistry.get_client(provider="disabled")
        
    return VideoService(s3_tool, llm_client, event_bus)

def get_event_service(event_bus: EventBus = Depends(get_event_bus)) -> EventService:
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    return EventService(event_bus)

def get_report_service(event_bus: EventBus = Depends(get_event_bus)) -> ReportService:
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    return ReportService(event_bus)
