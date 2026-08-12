import os

from pydantic import BaseModel, Field
from typing import Any

class Settings(BaseModel):
    # Application Mode: native, docker, production
    mode: str = Field(default_factory=lambda: os.getenv("MODE", "native"))
    model_free: bool = Field(default_factory=lambda: os.getenv("MODEL_FREE", "false").lower() == "true")

    # Infrastructure Backends
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql+asyncpg://vista:secret@localhost:5432/vista"))
    vector_backend_url: str = Field(default_factory=lambda: os.getenv("VECTOR_BACKEND_URL", "http://localhost:19530"))
    storage_backend_url: str = Field(default_factory=lambda: os.getenv("STORAGE_BACKEND_URL", "http://localhost:9000"))
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # Observability
    telemetry_exporter: str = Field(default_factory=lambda: os.getenv("TELEMETRY_EXPORTER", "console")) # console, otlp, none
    otlp_endpoint: str = Field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
    service_name: str = Field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "vista_agentic_ai"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    json_logs: bool = Field(default_factory=lambda: os.getenv("JSON_LOGS", "true").lower() == "true")
    
    # Frontend
    frontend_urls: str = Field(default_factory=lambda: os.getenv("FRONTEND_URLS", "http://localhost:5500,http://127.0.0.1:5500,http://[::]:5500"))
    
    # Enable toggles
    enable_tracing: bool = Field(default_factory=lambda: os.getenv("ENABLE_TRACING", "true").lower() == "true")
    enable_metrics: bool = Field(default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == "true")
    
    # Environment
    environment: str = Field(default_factory=lambda: os.getenv("VISTA_ENV", "development"))

    def model_post_init(self, __context: Any) -> None:
        if self.environment == "production":
            if "vista:secret@" in self.database_url:
                raise ValueError("CRITICAL: Default database password cannot be used in production")
            if "*" in self.frontend_urls:
                raise ValueError("CRITICAL: Wildcard CORS is not permitted in production")

config = Settings()
