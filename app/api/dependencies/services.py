from fastapi import Depends

from app.api.dependencies.repositories import (
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
from app.services.video_service.vlm_adapter import GeminiAdapter
from app.tools.video.s3_tool import S3Tool


def get_metadata_service(
    camera_repo: CameraRepository = Depends(get_camera_repository),
    alert_repo: AlertRepository = Depends(get_alert_repository),
    event_bus: EventBus = Depends(get_event_bus)
) -> MetadataService:
    return MetadataService(camera_repo, alert_repo, event_bus)

def get_vector_service(
    person_repo: PersonRepository = Depends(get_person_repository),
    vehicle_repo: VehicleRepository = Depends(get_vehicle_repository),
    event_bus: EventBus = Depends(get_event_bus)
) -> VectorService:
    return VectorService(person_repo, vehicle_repo, event_bus)

def get_s3_tool(event_bus: EventBus = Depends(get_event_bus)) -> S3Tool:
    return S3Tool(event_bus)

def get_video_service(
    s3_tool: S3Tool = Depends(get_s3_tool),
    event_bus: EventBus = Depends(get_event_bus)
) -> VideoService:
    return VideoService(s3_tool, GeminiAdapter(), event_bus)

def get_event_service(event_bus: EventBus = Depends(get_event_bus)) -> EventService:
    return EventService(event_bus)

def get_report_service(event_bus: EventBus = Depends(get_event_bus)) -> ReportService:
    return ReportService(event_bus)
