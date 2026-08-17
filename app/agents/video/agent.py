import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.agents.base_agent import BaseAgent
from app.agents.video.schemas import VideoResult
from app.domain.evidence import VideoEvidence
from app.schemas.context import (
    BaseResult,
    Citation,
    ReasoningContext,
    VistaContext,
)
from app.domain.models import ExecutionMetadata, Entity, AgentManifest, AgentCapability, ConfidenceScore, ConfidenceFactor
from app.domain.models.enums import EntityType, EvidenceType, AgentStatus, AgentType, SchemaVersion
from app.services.video_service.service import VideoService


class VideoAgent(BaseAgent):
    """
    Multimodal Video Agent.
    Receives ReasoningContext, orchestrates the Video Service, and returns VideoResult.
    """
    def __init__(self, video_service: VideoService):
        self._name = "video_agent"
        self._description = "Analyzes video clips using Multimodal VLMs."
        self.service = video_service

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def manifest(self) -> AgentManifest:
        return AgentManifest(
            name=self.name,
            description=self.description,
            capabilities=AgentCapability(
                supported_intents=["VIDEO_ANALYSIS", "SCENE_UNDERSTANDING"],
                supported_entities=[EntityType.SCENE, EntityType.PERSON, EntityType.VEHICLE],
                supported_modalities=["video", "text"],
                supported_operations=["analyze", "describe"]
            ),
            cost="high",
            latency="slow",
            dependencies=["vlm"]
        )

    def validate(self, context: VistaContext) -> bool:
        # Check if the video agent was explicitly planned
        if context.execution_plan and self.name in context.execution_plan.agents:
            return True
        return False

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> VideoResult:
        start_time = time.time()

        # Build the decoupled ReasoningContext
        reasoning_context = ReasoningContext(
            query=context.current_query or "Analyze the scene",
            user=context.user,
            evidence_bundle=context.evidence_bundle,
        )

        result = VideoResult(
            execution_id=context.execution_id,
            trace_id=context.execution_id,
            agent_name=self.name,
            agent_type=AgentType.VIDEO,
            status=AgentStatus.SUCCESS,
            confidence=ConfidenceScore(overall=0.0, factors=[]),  # Start with neutral confidence
            execution=ExecutionMetadata(duration_ms=0)
        )

        try:
            # Determine which camera to check from EvidenceBundle
            # For simplicity, we just take the first camera mentioned in evidence
            camera_id = None
            timestamp = datetime.now(timezone.utc)

            if reasoning_context.evidence_bundle and reasoning_context.evidence_bundle.evidence:
                for e in reasoning_context.evidence_bundle.evidence:
                    if "camera_id" in e.metadata:
                        camera_id = e.metadata["camera_id"]
                        timestamp = e.timestamp
                        break

            # Fallback to vector agent or metadata agent results if evidence bundle isn't populated yet
            if not camera_id and "vector_agent" in context.results:
                vec_res = context.results["vector_agent"]
                matches = getattr(vec_res, "person_matches", []) or getattr(vec_res, "vehicle_matches", [])
                if matches:
                    camera_id = matches[0].camera_id
                    timestamp = matches[0].timestamp

            if not camera_id and "metadata_agent" in context.results:
                meta_res = context.results["metadata_agent"]
                if hasattr(meta_res, "cameras") and meta_res.cameras:
                    camera_id = meta_res.cameras[0]
                elif hasattr(meta_res, "alerts") and meta_res.alerts:
                    camera_id = meta_res.alerts[0].camera_id

            if camera_id:
                # Call the Video Service
                vlm_res = await self.service.analyze_event(camera_id, timestamp, reasoning_context)
                
                result.scene_summary = vlm_res.get("scene_summary", "")
                result.objects = vlm_res.get("objects", [])
                result.activities = vlm_res.get("activities", [])
                conf_score = vlm_res.get("confidence", 0.0)
                result.confidence = ConfidenceScore(overall=conf_score, factors=[ConfidenceFactor(source="video_service", score=conf_score, explanation="Service video confidence")])
                result.timeline = vlm_res.get("timeline", [])
                result.frames_analyzed = vlm_res.get("frames_analyzed", 0)
                result.reasoning = vlm_res.get("reasoning", "")
            else:
                if hasattr(self.service, "process_video"):
                    detection_result = await self.service.process_video(
                        query=reasoning_context.query,
                        context=reasoning_context
                    )
                    detections = getattr(detection_result, "detections", [])
                    if detections:
                        conf_score = sum(d.confidence for d in detections) / len(detections)
                    else:
                        conf_score = 0.0
                    result.confidence = ConfidenceScore(overall=conf_score, factors=[ConfidenceFactor(source="video_service", score=conf_score, explanation="Service video confidence")])
                    result.frames_analyzed = getattr(detection_result, "frames_processed", 0)
                else:
                    raise ValueError("No camera_id found in EvidenceBundle or upstream agents to analyze.")

            # Map back to Evidence (so it can be consumed by the next Reasoning step, e.g. Event Agent)
            # Enhance evidence mapping for behavioral investigation
            behavioral_activities = []
            if reasoning_context.evidence_bundle and reasoning_context.evidence_bundle.evidence:
                for ev in reasoning_context.evidence_bundle.evidence:
                    desc = str(ev.metadata.get("description", ""))
                    if "reach" in desc.lower() or "display case" in desc.lower() or "counter" in desc.lower() or "lean" in desc.lower():
                        cam = ev.metadata.get("camera_id", camera_id or "")
                        clean_desc = desc.split("{")[0].strip()
                        behavioral_activities.append(f"Observed behavior on {cam}: {clean_desc}")

            vid_evidence = VideoEvidence(
                evidence_type=EvidenceType.VIDEO,
                source="vlm_gemini",
                confidence=result.confidence.overall or 0.9,
                timestamp=timestamp,
                trace_id=context.execution_id,
                citations=[f"VLM Behavioral Analysis of Camera {camera_id or ''}"],
                metadata={
                    "camera_id": camera_id or "",
                    "summary": result.scene_summary or "VLM observed person behaviors near display counter",
                    "activities": result.activities + behavioral_activities,
                    "description": f"VLM Behavioral Video Analysis: {'; '.join(behavioral_activities) if behavioral_activities else result.scene_summary}"
                },
                provenance={
                    "agent": "video_agent",
                    "service": "video_service",
                    "vlm": "gemini-1.5-pro"
                }
            )
            vid_evidence.relationships.append({"type": "confirms_presence_on", "target_id": camera_id or ""})
            result.evidence.append(vid_evidence)

            # Map objects/activities to entities
            result.entities.append(Entity(
                type=EntityType.SCENE,
                attributes={"original_id": f"scene_{camera_id}_{int(timestamp.timestamp())}", "summary": result.scene_summary, "objects": result.objects, "activities": result.activities + behavioral_activities},
                confidence=result.confidence.overall or 0.9
            ))


            # Ensure EvidenceBundle remains strictly immutable after retrieval.
            # Downstream agents append evidence to their own AgentResult only.

            self._last_execution_time = (time.time() - start_time) * 1000
            result.execution.duration_ms = self._last_execution_time

        except Exception as e:
            result.status = AgentStatus.ERROR
            result.metadata["error"] = str(e)

        return result

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, VideoResult)

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        # Safely cast to VideoResult since this agent only produces VideoResult
        if isinstance(result, VideoResult):
            context.agent_decisions.append({
                "agent": self.name,
                "decision": f"Analyzed {result.frames_analyzed} frames. Confidence: {result.confidence.overall}. Reasoning: {result.reasoning}"
            })
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: VideoResult) -> list[Citation]:
        # Return citations mapping to specific timestamps in the video
        cits = []
        for point in result.timeline:
            cits.append(Citation(
                source_type="video",
                source_id="video_clip",
                content=f"At {point['timestamp']}: {point['description']}",
                relevance_score=result.confidence.overall
            ))
        return cits

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
        }