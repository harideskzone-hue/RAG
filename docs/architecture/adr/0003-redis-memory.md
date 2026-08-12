# ADR 0003: Redis for Memory Checkpointing

## Status
**Accepted**

## Context
VISTA AI processes long-running asynchronous workflows, particularly around Video Analysis and Report Generation. If a worker node goes down during a 10-minute video analysis job, we need to recover the state. LangGraph maintains state in memory by default, which is not suitable for distributed scale-out.

## Alternatives Considered
- **PostgreSQL**: Slower for rapid state updates. Better suited for our static metadata (Cameras, Alerts).
- **In-Memory (RAM)**: Fails in multi-replica Docker/K8s deployments.
- **MongoDB**: Adds another infrastructural dependency, but we already have Redis for Pub/Sub.

## Decision
We chose the **Official `langgraph-checkpoint-redis`** library. It seamlessly connects LangGraph's state machine to a persistent Redis cluster.

## Consequences
- **Positive**: Blazing fast state updates. Built-in support for LangGraph graph versioning. Easily scales horizontally.
- **Negative**: State serialization requires strict typing (Pydantic). Non-serializable objects (like open network connections or Thread objects) cannot be stored in the graph state.
