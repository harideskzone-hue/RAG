import os
from typing import Any

from app.platform.config.config import config

import asyncio
from app.tools.metadata.store import get_metadata_store
from app.tools.vector.store import get_vector_store
from app.tools.video.store import get_blob_store

async def get_health() -> dict[str, Any]:
    db_store = get_metadata_store()
    vec_store = get_vector_store()
    blob_store = get_blob_store()

    db_status = "healthy" if await db_store.health() else "unhealthy"
    vec_status = "healthy" if await vec_store.health() else "unhealthy"
    blob_status = "healthy" if await blob_store.health() else "unhealthy"

    components = {
        "database": {"engine": "postgres" if config.mode != "native" else "mock_postgres", "status": db_status},
        "vector_store": {"engine": "milvus" if config.mode != "native" else "mock_milvus", "status": vec_status},
        "storage": {"engine": "s3" if config.mode != "native" else "mock_s3", "status": blob_status},
        "redis": {"engine": "redis" if config.mode != "native" else "in_memory", "status": "healthy"},
        "llm": {"engine": "gemini", "status": "configured" if config.mode == "native" or "REPLACE" not in os.getenv("GEMINI_API_KEY", "") else "needs_key"}
    }
    all_healthy = all(c["status"] in ["healthy", "configured"] for c in components.values())
    return {
        "status": "ok" if all_healthy else "degraded",
        "mode": config.mode,
        "components": components,
        "telemetry": config.telemetry_exporter
    }

async def get_ready() -> dict[str, Any]:
    health_info = await get_health()
    is_ready = health_info["status"] == "ok"
    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": health_info["components"]
    }

def get_live() -> dict[str, Any]:
    return {"status": "alive"}

def get_version() -> dict[str, str]:
    return {"version": "1.0.0"}

def get_build() -> dict[str, str]:
    return {"commit": "unknown", "date": "unknown"}

def get_info() -> dict[str, Any]:
    return {
        "service": "vista_agentic_ai",
        "description": "VISTA AI Orchestration Platform"
    }
