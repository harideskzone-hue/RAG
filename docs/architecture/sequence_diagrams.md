# Sequence Diagrams

These sequence diagrams illustrate the system interactions for various complex user queries.

## 1. Complex Reasoning Flow (Person Search + Metadata)

This flow occurs when a user asks: *"Find the person wearing a red backpack and check if the camera they are on is online."*

```mermaid
sequenceDiagram
    participant User
    participant App as FastAPI Layer
    participant Sup as Supervisor
    participant Plan as Planner
    participant EventBus as Event Bus
    participant Vec as Vector Agent
    participant Meta as Metadata Agent
    participant Evid as Evidence Agent
    participant Conf as Confidence Engine

    User->>App: POST /api/v1/chat "Find person with red backpack..."
    App->>Sup: run(context)
    
    Sup->>Plan: Generate Plan
    Plan-->>Sup: Plan(Intent=Search, Tasks=[VectorSearch, MetaCheck])
    
    Sup->>EventBus: Publish TASK_START
    
    par Parallel Execution
        EventBus->>Vec: TASK_START (VectorSearch)
        Vec->>Milvus: similarity_search()
        Milvus-->>Vec: Matches [cam_5]
        Vec->>EventBus: Publish VECTOR_RESULT
        
        EventBus->>Meta: TASK_START (MetaCheck)
        Meta->>PostgreSQL: select status from cameras where id=cam_5
        PostgreSQL-->>Meta: Status [online]
        Meta->>EventBus: Publish METADATA_RESULT
    end
    
    EventBus->>Evid: VECTOR_RESULT
    EventBus->>Evid: METADATA_RESULT
    
    Note over Evid: Deduplicate & align temporally
    Evid-->>Sup: EvidenceBundle generated
    
    Sup->>Conf: Evaluate EvidenceBundle
    Conf-->>Sup: ConfidenceScore & Reasoning
    
    Sup-->>App: Final Result
    App-->>User: "Found person on cam_5. Camera is online."
```

## 2. Long-Running Video Reasoning Flow

When analyzing video, the operation is asynchronous and uses WebSockets.

```mermaid
sequenceDiagram
    participant Client
    participant WS as WebSocket API
    participant Sup as Supervisor
    participant Vid as Video Agent
    participant Mem as Redis Memory
    
    Client->>WS: Connect WSS
    WS-->>Client: Connection Ack
    Client->>WS: "Analyze past 2 hours for fights"
    
    WS->>Sup: run_async(context)
    Sup->>Mem: Checkpoint State (PENDING)
    WS-->>Client: "Task Accepted. Generating..."
    
    Sup->>Vid: Start Video Analysis
    
    loop Every Chunk
        Vid->>S3: Download Chunk
        Vid->>VLM: Analyze Chunk
        Vid->>Mem: Update Progress
        Vid->>WS: Stream Progress Event
        WS-->>Client: "Processed chunk 5/20..."
    end
    
    Vid-->>Sup: Video Analysis Complete
    Sup->>Mem: Checkpoint State (COMPLETE)
    Sup->>WS: Stream Final Result
    WS-->>Client: Final Response with Clips
```
