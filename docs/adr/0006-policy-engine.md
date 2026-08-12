# ADR 0006: Policy Engine

## Problem
As the agentic system scales, it requires hard boundaries on cost, API rate limits, tool permissions, and operational constraints. Baking these rules into the Supervisor or Planner creates unmanageable spaghetti code and makes the system untestable.

## Decision
We introduced a strict `PolicyEngine`. It operates as a gatekeeper immediately after a plan is generated and before execution. It estimates costs, evaluates rules against a `PolicyContext`, resolves conflicts, and outputs a `PolicyDecision` (APPROVE, REJECT, MODIFY, DEFER) along with a `PolicyExplanation`.

## Alternatives Considered
- LLM-based policy checking (too slow, non-deterministic, and unsafe for hard financial/security boundaries).
- Simple `if/else` checks in the orchestrator (doesn't scale, hard to audit).

## Consequences
- **Positive**: Extreme auditability. We can trace exactly why a plan was pruned or rejected. The domain models (YAML/Pydantic) are authoritative and deterministic.
- **Negative**: Planners must be aware that their DAGs might be modified or rejected, requiring graceful fallback mechanisms.
