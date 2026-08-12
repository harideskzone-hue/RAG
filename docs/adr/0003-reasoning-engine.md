# ADR 0003: Reasoning Engine

## Problem
After specialized agents (Video, OCR, Metadata) execute, their disjointed findings must be synthesized into a cohesive conclusion. If the supervisor attempts this, it conflates orchestration with cognition.

## Decision
We abstracted synthesis into a dedicated `ReasoningEngine`. This component sits at the end of an investigation iteration. It ingests the `InvestigationMemory`, `KnowledgeGraph` context, and current `AgentResult` objects, and produces a structured `ReasoningResult` containing a hypothesis, missing evidence gaps, and `next_actions`.

## Alternatives Considered
- Merging reasoning into the Supervisor (violates single-responsibility principle).
- Relying on the initial Planner to do all reasoning (impossible since the Planner doesn't have the execution context yet).

## Consequences
- **Positive**: The Reasoning Engine can be swapped out for a more advanced VLM/LLM without touching the execution pipeline. It provides explicit "next_actions" that seamlessly feed back into the execution loop.
- **Negative**: Introduces a high-latency inference step at the end of every iterative cycle.
