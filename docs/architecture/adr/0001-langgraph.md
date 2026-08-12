# ADR 0001: LangGraph for Agent Orchestration

## Status
**Accepted**

## Context
VISTA AI is a multi-agent system. We need an orchestration framework to manage the lifecycle, state, and coordination of agents (Metadata, Vector, Video, Report). Standard rigid pipelines (like standard LLM Chains) cannot dynamically adapt to complex user queries that require loops or conditionally skipping certain tools.

## Alternatives Considered
- **LangChain (Standard Chains)**: Too linear. Difficult to implement dynamic loops (e.g., repeating a video search if the first chunk fails).
- **AutoGPT / BabyAGI (Autonomous Loops)**: Too unpredictable for enterprise use. We need bounded determinism.
- **Custom State Machine**: Requires building checkpointing, persistence, and graph traversal algorithms from scratch.

## Decision
We chose **LangGraph**. It provides:
1. Native support for cyclic graphs (loops).
2. Built-in state persistence (`MemorySaver`).
3. Predictable bounded execution (max steps limits).
4. First-class support for multi-agent Supervisor patterns.

## Consequences
- **Positive**: We inherit robust state management and debugging tools (LangSmith integration).
- **Negative**: Adds a layer of complexity; developers must understand graph concepts and state reducers rather than just linear functions.
