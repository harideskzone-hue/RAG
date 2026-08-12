import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.agents.base_agent import BaseAgent
from app.agents.metadata.schemas import MetadataResult
from app.agents.vector.schemas import VectorResult
from app.schemas.context import BaseResult, VistaContext
from app.services.metadata_service import MetadataService
from app.services.vector_service import VectorService
from app.domain.models import (
    AgentManifest,
    AgentCapability,
    EntityType,
    EvidenceType,
    AgentStatus,
    AgentType,
)
from app.domain.models.confidence import ConfidenceScore, ConfidenceFactor
from app.domain.evidence import EvidenceBundle, MetadataEvidence, PersonEvidence, VehicleEvidence
from app.agents.evidence.schemas import EvidenceResult


class EvidenceAgent(BaseAgent):
    """
    Evidence Agent.
    Collects and organizes evidence from various sources.
    """
    def __init__(self, metadata_service: MetadataService, vector_service: VectorService | None = None):
        self._name = "evidence_agent"
        self._description = "Collects and organizes evidence from metadata and vector stores."
        self.metadata_service = metadata_service
        self.vector_service = vector_service

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
                supported_intents=[
                    "CAMERA_STATUS",
                    "PERSON_SEARCH",
                    "VEHICLE_SEARCH",
                    "EVENT_SEARCH",
                    "REPORT",
                ],
                supported_entities=[EntityType.CAMERA, EntityType.ALERT, EntityType.PERSON, EntityType.VEHICLE],
                supported_modalities=["text", "metadata", "vector"],
                supported_operations=["collect", "organize"],
            ),
            cost="medium",
            latency="medium",
            dependencies=["metadata_agent", "vector_agent"],
        )

    def validate(self, context: VistaContext) -> bool:
        return self.name in (context.execution_plan.agents if context.execution_plan else [])

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> EvidenceResult:
        start_time = time.time()
        bundle = EvidenceBundle()

        try:
            # 1. Process Metadata
            if "metadata_agent" in context.results:
                meta_res: MetadataResult = context.results["metadata_agent"]

                for cam in meta_res.cameras:
                    if context.user and context.user.allowed_cameras is not None and cam.id not in context.user.allowed_cameras:
                        context.results["unauthorized_evidence_found"] = True
                        continue
                    evidence = MetadataEvidence(
                        evidence_id=str(uuid4()),
                        source="postgres_metadata",
                        confidence=0.95,  # High confidence for deterministic metadata, but not hardcoded to 1.0
                        timestamp=datetime.now(timezone.utc),
                        trace_id=context.execution_id,
                        citations=[f"Camera {cam.id}"],
                        metadata={"camera_id": cam.id, "location": cam.location, "description": f"Camera {cam.id} is {cam.status}"},
                        provenance={
                            "agent": "metadata_agent",
                            "service": "metadata_service",
                            "repository": "camera_repository",
                            "tool": "postgres_tool"
                        }
                    )
                    bundle.add_evidence(evidence)

                for alert in meta_res.alerts:
                    if context.user and context.user.allowed_cameras is not None and alert.camera_id not in context.user.allowed_cameras:
                        context.results["unauthorized_evidence_found"] = True
                        continue
                    evidence = MetadataEvidence(
                        evidence_id=str(uuid4()),
                        source="postgres_metadata",
                        confidence=0.95,  # High confidence for deterministic metadata, but not hardcoded to 1.0
                        timestamp=alert.timestamp,
                        trace_id=context.execution_id,
                        citations=[f"Alert {alert.id}"],
                        metadata={"camera_id": alert.camera_id, "alert_type": alert.type, "description": f"{alert.severity} alert: {alert.type}"},
                        provenance={
                            "agent": "metadata_agent",
                            "service": "metadata_service",
                            "repository": "alert_repository",
                            "tool": "postgres_tool"
                        }
                    )
                    bundle.add_evidence(evidence)

            # 2. Process Vector Matches
            if "vector_agent" in context.results:
                vec_res: VectorResult = context.results["vector_agent"]

                for person in vec_res.person_matches:
                    if context.user and context.user.allowed_cameras is not None and person.camera_id not in context.user.allowed_cameras:
                        context.results["unauthorized_evidence_found"] = True
                        continue
                    evidence = PersonEvidence(
                        evidence_id=str(uuid4()),
                        source="milvus_vector",
                        confidence=person.score,  # Raw Milvus score, reasoning engine will weight this later
                        timestamp=person.timestamp,
                        trace_id=context.execution_id,
                        citations=[f"Person Match {person.id}"],
                        metadata={"camera_id": person.camera_id, "description": person.description},
                        provenance={
                            "agent": "vector_agent",
                            "service": "vector_service",
                            "repository": "person_repository",
                            "tool": "milvus_tool"
                        }
                    )
                    # Relationship mapping example
                    evidence.relationships.append({"type": "appears_on", "target_id": person.camera_id})

                    bundle.add_evidence(evidence)

                for vehicle in vec_res.vehicle_matches:
                    if context.user and context.user.allowed_cameras is not None and vehicle.camera_id not in context.user.allowed_cameras:
                        context.results["unauthorized_evidence_found"] = True
                        continue
                    evidence = VehicleEvidence(
                        evidence_id=str(uuid4()),
                        source="milvus_vector",
                        confidence=vehicle.score,  # Raw Milvus score, reasoning engine will weight this later
                        timestamp=vehicle.timestamp,
                        trace_id=context.execution_id,
                        citations=[f"Vehicle Match {vehicle.id}"],
                        metadata={"camera_id": vehicle.camera_id, "description": vehicle.description},
                        provenance={
                            "agent": "vector_agent",
                            "service": "vector_service",
                            "repository": "vehicle_repository",
                            "tool": "milvus_tool"
                        }
                    )
                    # Relationship mapping example
                    evidence.relationships.append({"type": "appears_on", "target_id": vehicle.camera_id})

                    bundle.add_evidence(evidence)

            self._last_execution_time = (time.time() - start_time) * 1000

            if bundle.evidence:
                avg_conf = sum(item.confidence for item in bundle.evidence) / len(bundle.evidence)
            else:
                avg_conf = 0.5
            confidence_score = ConfidenceScore(overall=avg_conf, factors=[ConfidenceFactor(source="evidence_bundler", score=avg_conf, explanation="Average confidence of collected evidence")])
            return EvidenceResult(success=True, bundle=bundle, confidence=confidence_score)

        except Exception as e:
            # Set confidence based on success - Evidence collection is deterministic
            print(f"EvidenceAgent error: {e}")
            import traceback
            traceback.print_exc()
            confidence_score = ConfidenceScore(overall=0.0, factors=[])
            return EvidenceResult(success=False, error=str(e), bundle=bundle, confidence=confidence_score)

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, EvidenceResult)

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        if isinstance(result, EvidenceResult):
            # ResultCollector in Supervisor handles merging, but we fulfill the BaseAgent contract
            context.agent_decisions.append({
                "agent": self.name,
                "decision": f"Collected {len(result.bundle.evidence) if result.bundle else 0} evidence items.",
            })
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: BaseResult) -> list:
        return []

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
            "tokens": 0,
            "tool_latency": 0.0,
            "memory_usage": 0.0,
            "errors": 0,
            "retry_count": 0,
        }