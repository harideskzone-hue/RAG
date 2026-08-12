# VISTA AI Documentation Index

Welcome to the comprehensive documentation for the VISTA AI platform. This directory contains detailed architectural specifications, operational guides, and API references.

## Core Documentation Sections

### 🏗️ Architecture
Understand how VISTA AI is designed, how data flows, and why certain decisions were made.
- [Architecture Overview](architecture/overview.md)
- [System Architecture](architecture/system_architecture.md)
- [Component Diagrams](architecture/component_diagrams.md)
- [Sequence Diagrams](architecture/sequence_diagrams.md)
- [Data Flow](architecture/data_flow.md)
- [Databases](architecture/databases.md)
- [Architecture Decision Records (ADR)](architecture/adr/0001-langgraph.md)

### 🤖 Agents
Deep dives into the responsibilities and behaviors of each specialized LangGraph agent.
- [Intent Agent](agents/intent.md)
- [Planner Agent](agents/planner.md)
- [Supervisor Agent](agents/supervisor.md)
- [Metadata Agent](agents/metadata.md)
- [Vector Agent](agents/vector.md)
- [Video Agent](agents/video.md)
- [Event Agent](agents/event.md)
- [Report Agent](agents/report.md)
- [Confidence Engine](agents/confidence.md)

### 💻 Development
Guides for local setup, testing, and contributing to the VISTA AI codebase.
- [Developer Guide](development/developer_guide.md)
- [Folder Structure](development/folder_structure.md)
- [Coding Standards](development/coding_standards.md)
- [Testing Guide](development/testing.md)
- [Contributing](development/contributing.md)

### 🔌 API Integration
Specifications for interacting with the VISTA AI platform programmatically.
- [REST API](api/rest_api.md)
- [WebSocket API](api/websocket.md)
- [Authentication](api/authentication.md)
- [Integration Examples](api/examples.md)

### ⚙️ Operations & Observability
Best practices for monitoring, tracing, and maintaining the system in production.
- [Observability](operations/observability.md)
- [Monitoring](operations/monitoring.md)
- [Troubleshooting Guide](operations/troubleshooting.md)
- [Production Checklist](operations/production_checklist.md)

### 🚀 Deployment
Topologies and infrastructure as code for deploying VISTA AI at scale.
- [Docker & Compose](deployment/docker_compose.md)
- [Kubernetes & Helm](deployment/kubernetes.md)
- [Scaling Strategy](deployment/scaling.md)
- [Disaster Recovery](deployment/disaster_recovery.md)

### 🛠️ Configuration & Prompts
- [Environment Variables](configuration/environment_variables.md)
- [LLM Prompts Overview](prompts/intent.md)

### 📊 Performance
- [Benchmark Results](performance/benchmark_results.md)
- [SLAs & Optimization](performance/sla.md)

### 📅 Releases
- [Project Roadmap](releases/roadmap.md)
- [Changelog](releases/changelog.md)