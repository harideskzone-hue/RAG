import time
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.media.cache import VideoCache
from app.media.clip_selector import ClipSelector
from app.media.frame_sampler import FrameSampler, SamplingPolicy
from app.media.preprocessor import Preprocessor
from app.domain.models.reasoning_context import ReasoningContext
from app.schemas.context import VistaContext, UserContext
from app.services.video_service.vlm_adapter import BaseVLM
from app.tools.base_tool import BaseTool


class VideoService:
    """
    Orchestrates the Media Pipeline (Clip Selection, Sampling, Preprocessing) and the VLM.
    """
    def __init__(self, s3_tool: BaseTool, vlm: BaseVLM, event_bus: EventBus):
        self.s3_tool = s3_tool
        self.vlm = vlm
        self.event_bus = event_bus
        self.clip_selector = ClipSelector()
        self.sampler = FrameSampler()
        self.preprocessor = Preprocessor()
        self.cache = VideoCache()

    async def analyze_event(self, camera_id: str, timestamp: datetime, context: ReasoningContext) -> dict[str, Any]:
        start_time = time.time()
        user_id = getattr(getattr(context, 'user', None), 'user_id', 'unknown')
        self._publish_event("SERVICE_START", "video_service", user_id, start_time)

        # Camera RBAC enforcement
        user = getattr(context, 'user', None)
        if context and user and hasattr(user, 'allowed_cameras'):
            allowed_cameras = user.allowed_cameras
            # If allowed_cameras is not empty, enforce RBAC
            if allowed_cameras and camera_id not in allowed_cameras:
                error_msg = f"Camera {camera_id} not in allowed cameras: {allowed_cameras}"
                latency_ms = (time.time() - start_time) * 1000

                self._publish_event("SERVICE_ERROR", "video_service", user_id, start_time, end_time=time.time(), error=error_msg)
                raise ValueError(error_msg)

        prompt = context.query

        # 1. Clip Selection
        start_clip, end_clip = self.clip_selector.select_clip_window(timestamp)
        duration = int((end_clip - start_clip).total_seconds())

        # 2. Check Cache
        cached_result = self.cache.get(camera_id, start_clip.isoformat(), duration, prompt)
        if cached_result:
            self._publish_event("CACHE_HIT", "video_service", context.user.user_id, start_time)
            return cached_result

        self._publish_event("CACHE_MISS", "video_service", context.user.user_id, start_time)

        try:
            # 3. Retrieve Video (S3 Tool)
            # We mock the VistaContext requirement for BaseTool here, or adapt it.
            # BaseTool takes VistaContext, but VideoService receives ReasoningContext.
            # We'll construct a minimal dict kwargs compatible with S3Tool.
            # Create a minimal VistaContext-like object for the S3 tool with required fields
            s3_tool_context = type('S3ToolContext', (), {
                'user': context.user,
                'conversation_id': str(uuid4())  # Generate a random conversation ID for tracing
            })()

            s3_result = await self.s3_tool.execute(s3_tool_context, camera_id=camera_id, start_time=start_clip.isoformat(), end_time=end_clip.isoformat())
            if not s3_result.success:
                raise ValueError(s3_result.error)

            video_uri = s3_result.video_uri

            # 4. Media Pipeline: Sampling
            frames = self.sampler.sample_frames(video_uri, SamplingPolicy.BALANCED, duration)

            # 5. Media Pipeline: Preprocessing
            clean_frames = self.preprocessor.preprocess_frames(frames)

            # 6. VLM Analysis
            vlm_start = time.time()
            vlm_result = await self.vlm.analyze(clean_frames, prompt)
            vlm_result["frames_analyzed"] = len(clean_frames)

            # Telemetry for VLM
            self._publish_event("VLM_COMPLETE", "video_service", context.user.user_id, vlm_start, end_time=time.time(), metadata={"frames": len(clean_frames)})

            # 7. Cache the result
            self.cache.set(camera_id, start_clip.isoformat(), duration, prompt, vlm_result)

            self._publish_event("SERVICE_COMPLETE", "video_service", context.user.user_id, start_time, end_time=time.time())
            return vlm_result

        except Exception as e:
            self._publish_event("SERVICE_ERROR", "video_service", context.user.user_id, start_time, end_time=time.time(), error=str(e))
            raise

    def _publish_event(self, event_type: str, agent_name: str, trace_id: str, start_time: float, end_time: float = None, error: str = None, metadata: dict = None):
        kwargs = {
            "agent_name": agent_name,
            "event_type": event_type,
            "start_time": start_time,
            "status": "ERROR" if error else "SUCCESS",
            "trace_id": trace_id,
            "metadata": metadata or {}
        }
        if end_time:
            kwargs["end_time"] = end_time
            kwargs["latency_ms"] = (end_time - start_time) * 1000
        if error:
            kwargs["errors"] = [error]

        self.event_bus.publish(AgentEvent(**kwargs))