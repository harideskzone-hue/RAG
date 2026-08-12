# ADR 0001: Agent Contract

## Problem
In a multi-agent system, heterogeneous agents (Video, OCR, DB, RAG) require a standardized way to communicate their inputs, processing results, and confidence levels back to the orchestrator. Without a strict contract, parsing responses becomes brittle and scaling the number of agents introduces technical debt.

## Decision
We enforce a strict `AgentManifest` and `AgentResult` contract. Every agent must declare its capabilities, required inputs, and SLA via a manifest. Every agent must return exactly one format: an `AgentResult` object containing standard metadata, the typed payload, and a `confidence_score`.

## Alternatives Considered
- Returning raw strings (brittle, unstructured).
- Using loosely defined JSON objects per agent (creates O(N) parsers in the Supervisor).
- Standardizing on purely semantic "Tool Calls" (lacked metadata, SLA, and routing control).

## Consequences
- **Positive**: Adding a new agent requires zero changes to the orchestrator. Type safety is guaranteed.
- **Negative**: Agent developers must write boilerplate wrappers to conform to the contract.
