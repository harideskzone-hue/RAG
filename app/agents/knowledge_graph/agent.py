import time
from typing import Any
import uuid

from app.agents.base_agent import BaseAgent
from app.domain.evidence import EvidenceBundle, PersonEvidence, MetadataEvidence
from app.domain.knowledge_graph.node import Node
from app.domain.knowledge_graph.edge import Edge
from app.domain.knowledge_graph.updater import GraphUpdate
from app.domain.models import AgentManifest, AgentCapability
from app.schemas.context import VistaContext, BaseResult
from app.services.metadata_service import MetadataService
from app.domain.models import Entity, ExecutionMetadata, AgentManifest, AgentCapability, ConfidenceScore, ConfidenceFactor
from app.domain.models.enums import EntityType, EvidenceType, AgentStatus, AgentType, SchemaVersion
from app.domain.evidence import MetadataEvidence


class GraphResult(BaseResult):
    nodes: int = 0
    edges: int = 0


class KnowledgeGraphAgent(BaseAgent):
    """
    Knowledge Graph Agent.
    Transforms an EvidenceBundle into a GraphUpdate to persist strict provenance.
    """
    def __init__(self):
        self._name = "knowledge_graph_agent"
        self._description = "Translates EvidenceBundle into Knowledge Graph nodes and edges while preserving provenance."
        self._last_execution_time = 0.0

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
                supported_intents=["KNOWLEDGE_GRAPH_UPDATE"],
                supported_entities=[],
                supported_modalities=["evidence_bundle"],
                supported_operations=["graph_update"]
            ),
            cost="low",
            latency="fast"
        )

    def validate(self, context: VistaContext) -> bool:
        return context.evidence_bundle is not None

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any = None) -> GraphResult:
        start_time = time.time()
        bundle: EvidenceBundle = context.evidence_bundle

        new_nodes = []
        new_edges = []

        for ev in bundle.evidence:
            if isinstance(ev, MetadataEvidence):
                # Camera node
                cam_id = ev.metadata.get("camera_id", "UNKNOWN_CAM")
                node = Node(
                    id=uuid.uuid5(uuid.NAMESPACE_OID, cam_id),
                    type="Camera",
                    attributes={
                        "camera_id": cam_id,
                        "location": ev.metadata.get("location", ""),
                        "source_evidence_id": str(ev.evidence_id),
                        "timestamp": ev.timestamp.isoformat()
                    },
                    confidence=ev.confidence,
                    created_at=ev.created_at.timestamp()
                )
                new_nodes.append(node)

            elif isinstance(ev, PersonEvidence):
                # Person node
                person_id = uuid.uuid5(uuid.NAMESPACE_OID, f"person_{ev.evidence_id}")

                person_node = Node(
                    id=person_id,
                    type="Person",
                    attributes={
                        "description": ev.metadata.get("description", ""),
                        "source_evidence_id": str(ev.evidence_id),
                        "source_frame_id": ev.source_id or "unknown_frame",
                        "timestamp": ev.timestamp.isoformat()
                    },
                    confidence=ev.confidence,
                    created_at=ev.created_at.timestamp()
                )
                new_nodes.append(person_node)

                # Edge from Person to Camera
                cam_node_id = uuid.uuid5(uuid.NAMESPACE_OID, f"camera_{ev.metadata.get('camera_id', 'unknown')}")
                edge = Edge(
                    source_id=person_id,
                    target_id=cam_node_id,
                    type="appears_on",
                    attributes={
                        "evidence_id": str(ev.evidence_id),
                        "frame_id": ev.source_id or "unknown_frame",
                        "timestamp": ev.timestamp.isoformat(),
                        "confidence": ev.confidence
                    },
                    confidence=ev.confidence,
                    created_at=ev.created_at.timestamp()
                )
                new_edges.append(edge)

        graph_update = GraphUpdate(
            new_nodes=new_nodes,
            new_edges=new_edges,
            updated_nodes=[],
            deleted_nodes=[],
            deleted_edges=[]
        )

        from app.domain.knowledge_graph.repository import InMemoryRepository
        from app.domain.knowledge_graph.builder import GraphBuilder

        repo = InMemoryRepository()
        graph = repo.load("main_investigation")
        builder = GraphBuilder(graph)

        builder.apply_update(graph_update)
        repo.save("main_investigation", graph)

        # Set confidence based on success - graph construction is deterministic
        confidence_score = ConfidenceScore(overall=0.95, factors=[])
        return GraphResult(success=True, error=None, nodes=len(new_nodes), edges=len(new_edges), confidence=confidence_score)

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, GraphResult) and result.success

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        if isinstance(result, BaseResult):
            # ResultCollector in Supervisor handles merging, but we fulfill the BaseAgent contract
            context.results[self.name] = result
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