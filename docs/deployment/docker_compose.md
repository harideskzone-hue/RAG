# Docker Compose Deployment

The fastest way to spin up VISTA AI locally is using Docker Compose.

## Prerequisites
- Docker & Docker Compose plugin
- Minimum 16GB RAM

## Command
```bash
docker-compose up -d --build
```

## Services Started
1. `vista-api`: The FastAPI backend
2. `postgres`: Metadata DB
3. `milvus`: Vector DB (Standalone)
4. `redis`: State Checkpointing
5. `jaeger`: OpenTelemetry tracing UI
