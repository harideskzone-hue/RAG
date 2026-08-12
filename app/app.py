from fastapi import FastAPI

from app.api.exception_handlers import add_exception_handlers
from app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.middleware.tracing import TracingMiddleware
from app.api.v1.routes import chat, health, investigations, reports, websocket
from app.platform.telemetry.orchestrator import init_telemetry


def create_app() -> FastAPI:
    # Initialize Core Telemetry (Logging, Tracing, Metrics)
    init_telemetry()

    app = FastAPI(
        title="VISTA AI",
        description="VISTA AI Agentic Orchestration Platform",
        version="1.0.0"
    )

    # Add Exception Handlers
    add_exception_handlers(app)

    # Middleware Stack (Execution order is bottom-up in Starlette)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(TracingMiddleware)
    
    from fastapi.middleware.cors import CORSMiddleware
    from app.platform.config.config import config
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.frontend_urls.split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Execution-ID"],
    )
    
    # Routers
    api_v1_prefix = "/api/v1"
    app.include_router(chat.router, prefix=api_v1_prefix)
    app.include_router(reports.router, prefix=api_v1_prefix)
    app.include_router(investigations.router, prefix=api_v1_prefix)
    app.include_router(health.router, prefix=api_v1_prefix)
    app.include_router(websocket.router, prefix=api_v1_prefix)

    return app

app = create_app()
