import time
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

from app.agents.base_agent import BaseAgent
from app.agents.intent.enums import Intent
from app.agents.vector.schemas import VectorResult
from app.schemas.context import BaseResult, Citation, VistaContext
from app.services.vector_service import RetrievalMode, VectorService
from app.domain.models import Entity, ExecutionMetadata, AgentManifest, AgentCapability, ConfidenceScore, ConfidenceFactor
from app.domain.models.enums import EntityType, EvidenceType, AgentStatus, AgentType, SchemaVersion
from app.domain.evidence import PersonEvidence, VehicleEvidence

from app.agents.vector.expander import QueryExpander
from app.agents.vector.reranker import PassThroughReranker


class VectorAgent(BaseAgent):
    """
    Vector Agent.
    Interprets intent, expands query, calls the Vector Service concurrently, reranks, and formats results.
    Does NOT know about Milvus internals.
    """
    def __init__(self, vector_service: VectorService, encoder=None, llm_client=None):
        self._name = "vector_agent"
        self._description = "Performs semantic search for appearances, people, and vehicles."
        if vector_service is None or not hasattr(vector_service, "search_person"):
            from app.api.dependencies.services import get_vector_service
            self.service = get_vector_service()
        else:
            self.service = vector_service
        self._encoder = encoder
        self.expander = QueryExpander(llm_client=llm_client)
        self.reranker = PassThroughReranker()

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
                supported_intents=["PERSON_SEARCH", "VEHICLE_SEARCH"],
                supported_entities=[EntityType.PERSON, EntityType.VEHICLE],
                supported_modalities=["embedding", "image", "text"],
                supported_operations=["similarity_search"]
            ),
            cost="medium",
            latency="fast",
            dependencies=["milvus"]
        )

    def validate(self, context: VistaContext) -> bool:
        return context.execution_plan is not None and self.name in context.execution_plan.agents

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> VectorResult:
        start_time = time.time()
        intent = context.execution_plan.intent if context.execution_plan else "PERSON_SEARCH"
        intent_result = context.results.get("intent_agent")
        entities = intent_result.entities if intent_result else {}
        
        # Original search query
        original_query = entities.get("description", context.current_query)

        # Initialize with neutral confidence
        result = VectorResult(
            execution_id=context.execution_id,
            trace_id=context.execution_id,
            agent_name=self.name,
            agent_type=AgentType.VECTOR,
            status=AgentStatus.SUCCESS,
            confidence=ConfidenceScore(overall=0.0, factors=[]),
            execution=ExecutionMetadata(duration_ms=0)
        )

        try:
            # 1. Query Expansion
            expanded_queries = [original_query]
            if intent_result:
                try:
                    expanded = await self.expander.expand(intent_result)
                    if expanded:
                        expanded_queries = expanded
                except Exception as ex:
                    logger.warning(f"Query expansion failed: {ex}. Using original query.")
                
            if self._encoder is not None:
                encoder = self._encoder
            else:
                from app.tools.vector.encoder import get_vector_encoder
                encoder = get_vector_encoder()
                
            mode = RetrievalMode.BALANCED
            intent_lower = intent.lower() if intent else ""
            
            # Capability-based routing using search_operations dynamically produced by LLM
            query_intent = getattr(intent_result, 'query_intent', None)
            search_ops = set(getattr(query_intent, 'search_operations', []) or [])
            
            run_person_cap = "vector_person" in search_ops
            run_vehicle_cap = "vector_vehicle" in search_ops
            
            # If search_operations is unpopulated, derive capabilities from intent or run both for full coverage
            if not run_person_cap and not run_vehicle_cap:
                intent_val = getattr(intent_result, 'intent', None)
                intent_name = str(intent_val.value if hasattr(intent_val, 'value') else intent_val).lower()
                if intent_name == "vehicle_search":
                    run_vehicle_cap = True
                else:
                    run_person_cap = True

            person_candidates = []
            vehicle_candidates = []

            # 2. Parallel Retrieval based on active capabilities
            for query in expanded_queries:
                query_embedding = encoder.encode(query)
                if run_person_cap:
                    try:
                        p_res = await self.service.search_person(query_embedding, mode, context)
                        person_candidates.extend(p_res)
                    except Exception as err:
                        logger.error(f"search_person failed for query '{query}': {err}")
                if run_vehicle_cap:
                    try:
                        v_res = await self.service.search_vehicle(query_embedding, mode, context)
                        vehicle_candidates.extend(v_res)
                    except Exception as err:
                        logger.error(f"search_vehicle failed for query '{query}': {err}")

            # 3. Reranking
            if person_candidates:
                person_candidates = await self.reranker.rerank(original_query, person_candidates, context)
                result.person_matches = person_candidates
            if vehicle_candidates:
                vehicle_candidates = await self.reranker.rerank(original_query, vehicle_candidates, context)
                result.vehicle_matches = vehicle_candidates

            self._last_execution_time = (time.time() - start_time) * 1000
            result.execution.duration_ms = self._last_execution_time

            # 4. Map domain models to Evidence objects
            for match in result.person_matches:
                origin_dict = getattr(match, "origin", None) or {}
                attr_dict = getattr(match, "attributes", None) or {}
                ev_source = origin_dict.get("type", "vector_agent")
                
                desc = match.description or f"Person track {match.id} observed on camera {match.camera_id}"
                meta = {
                    "camera_id": match.camera_id,
                    "description": desc,
                    "bbox": match.bbox,
                    "origin": origin_dict,
                    "attributes": attr_dict
                }
                result.evidence.append(PersonEvidence(
                    evidence_type=EvidenceType.VECTOR,
                    source=ev_source,
                    confidence=match.score,
                    timestamp=match.timestamp,
                    trace_id=context.execution_id,
                    metadata=meta
                ))
                result.entities.append(Entity(
                    type=EntityType.PERSON,
                    attributes={"original_id": match.id, "description": match.description, "bbox": match.bbox, "origin": origin_dict, "attributes": attr_dict},
                    confidence=match.score
                ))

            for match in result.vehicle_matches:
                result.evidence.append(VehicleEvidence(
                    evidence_type=EvidenceType.VECTOR,
                    source="vector_agent",
                    confidence=match.score,
                    timestamp=match.timestamp,
                    trace_id=context.execution_id,
                    metadata={"camera_id": match.camera_id, "description": match.description, "license_plate": match.license_plate}
                ))
                result.entities.append(Entity(
                    type=EntityType.VEHICLE,
                    attributes={"original_id": match.id, "description": match.description, "license_plate": match.license_plate},
                    confidence=match.score
                ))

            # Update overall confidence based on top match score
            scores = [m.score for m in result.person_matches] + [m.score for m in result.vehicle_matches]
            if scores:
                top_score = max(scores)
                result.confidence = ConfidenceScore(overall=top_score, factors=[ConfidenceFactor(source="milvus", score=top_score, explanation="Top match score")])

        except Exception as e:
            result.status = AgentStatus.ERROR
            result.metadata["error"] = str(e)  # Sanitized! No traceback leak.

        return result

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, VectorResult)

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        if isinstance(result, VectorResult):
            context.agent_decisions.append({
                "agent": self.name,
                "decision": f"Retrieved {len(result.person_matches)} person matches and {len(result.vehicle_matches)} vehicle matches."
            })
        return context

    def confidence(self, result: BaseResult) -> float:
        if hasattr(result, 'person_matches') and hasattr(result, 'vehicle_matches'):
            scores = [m.score for m in result.person_matches] + [m.score for m in result.vehicle_matches]
            top_score = max(scores) if scores else 0.0
            if top_score > 0:
                result.confidence = ConfidenceScore(overall=top_score, factors=[ConfidenceFactor(source="milvus", score=top_score, explanation="Top match score")])
            return top_score
        return result.confidence.overall

    def citations(self, result: VectorResult) -> list[Citation]:
        citations = []
        for match in result.person_matches:
            citations.append(Citation(
                source_type="milvus",
                source_id=match.id,
                content=f"Vector Person Match (id={match.id}, score={match.score})",
                relevance_score=match.score
            ))
        return citations

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
            "tokens": 0,
            "tool_latency": 0.0,
            "memory_usage": 0.0,
            "errors": 0,
            "retry_count": 0
        }