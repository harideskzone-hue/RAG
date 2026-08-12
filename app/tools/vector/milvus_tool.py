import time
from typing import Any

from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool
from app.tools.vector.schemas import VectorToolResult
from app.tools.vector.store import get_vector_store
from app.platform.config.config import config


class MilvusTool(BaseTool):
    """
    Executes raw vector search against Milvus (or NativeVectorStore).
    Generic: Requires collection name and embedding.
    """
    def __init__(self, event_bus: EventBus):
        self._name = "milvus"
        self._description = "Executes similarity search on the vector database."
        self.event_bus = event_bus
        self.store = get_vector_store()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, context: VistaContext, **kwargs) -> VectorToolResult:
        collection = kwargs.get("collection", "")
        embedding = kwargs.get("embedding", [])
        top_k = kwargs.get("top_k", 5)
        
        start_time = time.time()
        
        # Publish telemetry start
        self.event_bus.publish(AgentEvent(
            agent_name="tool_milvus",
            event_type="TOOL_START",
            start_time=start_time,
            status="RUNNING",
            trace_id=context.conversation_id,
            metadata={"collection": collection, "top_k": top_k}
        ))
        
        try:
            allowed_cameras = getattr(context.user, "allowed_cameras", None) if getattr(context, "user", None) else None
            matches = await self.store.search(collection, embedding, top_k, allowed_cameras=allowed_cameras)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Publish telemetry success
            self.event_bus.publish(AgentEvent(
                agent_name="tool_milvus",
                event_type="TOOL_COMPLETE",
                start_time=start_time,
                end_time=time.time(),
                status="SUCCESS",
                latency_ms=latency_ms,
                trace_id=context.conversation_id,
                metadata={"matches_returned": len(matches)}
            ))
            
            return VectorToolResult(
                success=True,
                matches=matches,
                collection_searched=collection
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Publish telemetry error
            self.event_bus.publish(AgentEvent(
                agent_name="tool_milvus",
                event_type="TOOL_ERROR",
                start_time=start_time,
                end_time=time.time(),
                status="ERROR",
                latency_ms=latency_ms,
                errors=[str(e)],
                trace_id=context.conversation_id,
                metadata={"collection": collection}
            ))
            
            return VectorToolResult(
                success=False,
                error=str(e),
                collection_searched=collection
            )

    def validate(self, **kwargs) -> bool:
        return "collection" in kwargs and isinstance(kwargs["collection"], str) and "embedding" in kwargs

    async def health(self) -> bool:
        return await self.store.health()

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "vector_database",
            "engine": "milvus_or_native",
            "mocked": config.mode == "native"
        }
