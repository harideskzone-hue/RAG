# ADR 0002: Supervisor

## Problem
Complex investigations require multiple agents operating in iterative loops. Hardcoding workflow paths (e.g., Video -> OCR -> Reasoning) limits the AI's ability to pivot when new evidence is found. However, full LLM autonomy often leads to infinite loops and massive API costs.

## Decision
We chose a declarative Supervisor pattern operating strictly as an execution scheduler, not an intelligence layer. The Supervisor orchestrates based on a generated `ExecutionPlan`, executing agent groups asynchronously. It relies entirely on the `PolicyEngine` to mutate plans and manage budgets, and on the `ConfidenceAggregator` to determine when to stop.

## Alternatives Considered
- **LangChain/AutoGPT style autonomy**: (Rejected due to infinite loops and lack of auditability).
- **Static DAGs (Airflow style)**: (Rejected because investigations require dynamic paths).

## Consequences
- **Positive**: The Supervisor contains zero business logic, ensuring extreme stability. It behaves like an OS scheduler.
- **Negative**: The intelligence burden is shifted heavily onto the Planner and Policy Engine to produce valid DAGs.
