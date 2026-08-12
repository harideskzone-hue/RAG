import time
from typing import Any

from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool
from app.tools.video.schemas import VideoToolResult
from app.tools.video.store import get_blob_store
from app.platform.config.config import config


class S3Tool(BaseTool):
    """
    Retrieves video files from S3/blob storage (or LocalFileStore in native mode).
    """
    def __init__(self, event_bus: EventBus):
        self._name = "s3"
        self._description = "Retrieves raw video files from object storage."
        self.event_bus = event_bus
        self.store = get_blob_store()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, context: VistaContext, **kwargs) -> VideoToolResult:
        start_time = time.time()
        camera_id = kwargs.get("camera_id", "")
        start_time_iso = kwargs.get("start_time", "")
        end_time_iso = kwargs.get("end_time", "")

        # Camera RBAC enforcement
        if context and context.user and hasattr(context.user, 'allowed_cameras'):
            allowed_cameras = context.user.allowed_cameras
            # If allowed_cameras is not empty, enforce RBAC
            if allowed_cameras and camera_id not in allowed_cameras:
                error_msg = f"Camera {camera_id} not in allowed cameras: {allowed_cameras}"
                latency_ms = (time.time() - start_time) * 1000

                # Publish telemetry error
                self.event_bus.publish(AgentEvent(
                    agent_name="tool_s3",
                    event_type="TOOL_ERROR",
                    start_time=start_time,
                    end_time=time.time(),
                    status="ERROR",
                    latency_ms=latency_ms,
                    errors=[error_msg],
                    trace_id=context.conversation_id if context else "unknown_trace",
                    metadata={"camera_id": camera_id}
                ))

                return VideoToolResult(
                    success=False,
                    error=error_msg
                )

        start_time = time.time()

        trace_id = context.conversation_id if context else "unknown_trace"

        # Publish telemetry start
        self.event_bus.publish(AgentEvent(
            agent_name="tool_s3",
            event_type="TOOL_START",
            start_time=start_time,
            status="RUNNING",
            trace_id=trace_id,
            metadata={"camera_id": camera_id, "start": start_time_iso, "end": end_time_iso}
        ))

        try:
            bucket_name = "vista-video-bucket"
            object_key = f"{camera_id}_{start_time_iso}_{end_time_iso}.mp4"
            if config.mode == "native":
                object_key = "cctv.mp4"

            video_uri, size_mb = await self.store.get_uri(bucket_name, object_key)

            latency_ms = (time.time() - start_time) * 1000

            # Publish telemetry success
            self.event_bus.publish(AgentEvent(
                agent_name="tool_s3",
                event_type="TOOL_COMPLETE",
                start_time=start_time,
                end_time=time.time(),
                status="SUCCESS",
                latency_ms=latency_ms,
                trace_id=trace_id,
                metadata={"video_uri": video_uri}
            ))

            return VideoToolResult(
                success=True,
                video_uri=video_uri,
                metadata={"size_mb": size_mb}
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Publish telemetry error
            self.event_bus.publish(AgentEvent(
                agent_name="tool_s3",
                event_type="TOOL_ERROR",
                start_time=start_time,
                end_time=time.time(),
                status="ERROR",
                latency_ms=latency_ms,
                errors=[str(e)],
                trace_id=trace_id,
                metadata={"camera_id": camera_id}
            ))

            return VideoToolResult(
                success=False,
                error=str(e)
            )

    def validate(self, **kwargs) -> bool:
        return "camera_id" in kwargs and "start_time" in kwargs and "end_time" in kwargs

    async def health(self) -> bool:
        return await self.store.health()

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "blob_storage",
            "engine": "s3_or_local",
            "mocked": config.mode == "native"
        }