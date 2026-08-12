# Database Architecture

VISTA AI uses a polyglot persistence architecture to handle the distinct requirements of multimodal data.

## PostgreSQL
- **Purpose**: Relational metadata, RBAC, Camera Inventories, and deterministic Alerts.
- **Tables**: `cameras`, `alerts`, `users`, `roles`
- **Retention**: Persistent, long-term storage.

## Milvus
- **Purpose**: Dense vector storage for semantic search and visual embeddings (e.g. Person Search).
- **Collections**: `person_collection`, `vehicle_collection`.
- **Indexes**: IVF_FLAT, HNSW for rapid nearest-neighbor search.
- **Retention**: Ephemeral (7-30 days) depending on compliance.

## Redis
- **Purpose**: High-throughput distributed state management and memory checkpointing.
- **Keys**: `vista:state:{conversation_id}`
- **Retention**: Managed by the `MemoryManager` eviction policies.

## AWS S3
- **Purpose**: Unstructured raw video clips and evidence attachments.
- **Retention**: Lifecycle policies transition to Glacier after 30 days.
