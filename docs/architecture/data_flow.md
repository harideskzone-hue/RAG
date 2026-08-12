# Data Flow

The data flow within VISTA AI guarantees that raw infrastructure boundaries are completely isolated from reasoning boundaries.

```mermaid
graph TD
    User((User)) --> Intent
    
    subgraph Reasoning Layer
        Intent[Intent Agent] --> Planner[Planner Agent]
        Planner --> Supervisor[Supervisor]
    end
    
    subgraph Execution Layer
        Supervisor -->|Tasks| Agents(Metadata, Vector, Video)
    end
    
    subgraph Storage Layer
        Agents -->|Queries| DB[(PostgreSQL / Milvus / S3)]
        DB -->|Raw Results| Agents
    end
    
    subgraph Evidence Layer
        Agents -->|AgentEvents| Evidence[Evidence Agent]
        Evidence -->|EvidenceBundle| Confidence[Confidence Engine]
    end
    
    Confidence -->|Scored Response| Supervisor
    Supervisor --> User
```

1. **User Query**: Received via REST/WebSocket.
2. **Intent & Planning**: The orchestrator determines the optimal combination of tools.
3. **Execution**: The Supervisor dispatches tasks across the EventBus.
4. **Data Retrieval**: Domain agents query their respective databases.
5. **Evidence Alignment**: The Evidence agent subscribes to results, creating a unified temporal `EvidenceBundle`.
6. **Scoring**: The Confidence engine evaluates policies.
7. **Response**: Returned to the user.
