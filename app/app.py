import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

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
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routers
    api_v1_prefix = "/api/v1"
    app.include_router(chat.router, prefix=api_v1_prefix)
    app.include_router(reports.router, prefix=api_v1_prefix)
    app.include_router(investigations.router, prefix=api_v1_prefix)
    app.include_router(health.router)
    app.include_router(health.router, prefix=api_v1_prefix)
    app.include_router(websocket.router, prefix=api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/ui")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_dir = os.path.join(project_root, "dataset", "events")
    os.makedirs(events_dir, exist_ok=True)
    app.mount("/media/events", StaticFiles(directory=events_dir), name="events")

    persons_dir = os.path.join(project_root, "dataset", "persons")
    os.makedirs(persons_dir, exist_ok=True)
    app.mount("/media/persons", StaticFiles(directory=persons_dir), name="persons")

    tracks_dir = os.path.join(project_root, "dataset", "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    app.mount("/media/tracks", StaticFiles(directory=tracks_dir), name="tracks")

    input_dir = os.path.join(project_root, "input")
    os.makedirs(input_dir, exist_ok=True)
    app.mount("/media/videos", StaticFiles(directory=input_dir), name="videos")

    frontend_dir = os.path.join(project_root, "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")

    return app

app = create_app()
