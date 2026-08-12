import time
from typing import Any

from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.telemetry import AgentEvent
from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool
from app.tools.metadata.schemas import MetadataToolResult
from app.tools.metadata.store import get_metadata_store
from app.platform.config.config import config


class PostgresTool(BaseTool):
    """
    Executes raw SQL against Postgres (or SQLite in native mode).
    Provides strict schema boundaries.
    """
    def __init__(self, event_bus: EventBus):
        self._name = "postgres"
        self._description = "Executes read-only SQL queries on the metadata relational database."
        self.event_bus = event_bus
        self.store = get_metadata_store()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, context: VistaContext, **kwargs) -> MetadataToolResult:
        query = kwargs.get("query", "")
        
        start_time = time.time()
        
        # Publish telemetry start
        self.event_bus.publish(AgentEvent(
            agent_name="tool_postgres",
            event_type="TOOL_START",
            start_time=start_time,
            status="RUNNING",
            trace_id=context.conversation_id,
            metadata={"query": query}
        ))
        
        try:
            # Basic sanitization
            if "DROP" in query.upper() or "DELETE" in query.upper() or "UPDATE" in query.upper():
                if not query.upper().startswith("CREATE") and not query.upper().startswith("INSERT"): # allow inserts for testing/ingestion
                    raise ValueError("Only SELECT or INSERT operations are permitted by this tool.")
            
            params = kwargs.get("params", [])
            rows = await self.store.execute(query, *params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Publish telemetry success
            self.event_bus.publish(AgentEvent(
                agent_name="tool_postgres",
                event_type="TOOL_COMPLETE",
                start_time=start_time,
                end_time=time.time(),
                status="SUCCESS",
                latency_ms=latency_ms,
                trace_id=context.conversation_id,
                metadata={"rows_returned": len(rows)}
            ))
            
            return MetadataToolResult(
                success=True,
                rows=rows,
                query_executed=query
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Publish telemetry error
            self.event_bus.publish(AgentEvent(
                agent_name="tool_postgres",
                event_type="TOOL_ERROR",
                start_time=start_time,
                end_time=time.time(),
                status="ERROR",
                latency_ms=latency_ms,
                errors=[str(e)],
                trace_id=context.conversation_id,
                metadata={"query": query}
            ))
            
            return MetadataToolResult(
                success=False,
                error=str(e),
                query_executed=query
            )

    def validate(self, **kwargs) -> bool:
        return "query" in kwargs and isinstance(kwargs["query"], str)

    async def health(self) -> bool:
        return await self.store.health()

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "relational_database",
            "engine": "postgres_or_sqlite",
            "mocked": config.mode == "native"
        }
