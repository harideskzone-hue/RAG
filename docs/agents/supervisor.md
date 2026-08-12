# Supervisor Agent

## Purpose
The core orchestrator of VISTA AI. It manages the LangGraph state machine and delegates tasks across the EventBus.

## Inputs
- `VistaContext`: The global state object containing memory, user roles, and the current query.

## Outputs
- `VistaContext`: Mutated state with finalized `confidence_score` and `evidence_bundle`.

## Dependencies
- EventBus
- Planner Agent
- Confidence Engine

## Failure Behavior
Implements exponential backoff and localized retries. If a critical node (like Planner) fails, it gracefully degrades by returning partial results.
