# ADR 0007: Evaluation Framework

## Problem
Iterating on Agentic AI is dangerous without measurement. Changing a prompt in the Planner or adjusting a threshold in the Policy Engine might drastically degrade the system's ability to solve investigations. We need a way to detect regressions automatically.

## Decision
We built a deterministic and hybrid `EvaluationFramework`. It leverages declarative `EvaluationManifests` and `BenchmarkSuites`. We decoupled the evaluation logic from the agents entirely via an `EvaluationAdapter`. The system compares current runs against historical `EvaluationBaseline` data using specialized Subsystem Scorers (Planner, Graph, Memory, etc.).

## Alternatives Considered
- Evaluating end-to-end output only (fails to identify *which* subsystem caused the regression).
- Pure LLM-as-a-judge for everything (too expensive and unpredictable for strict logic checks like DAG validation).

## Consequences
- **Positive**: We can iterate aggressively with a safety net. CI/CD integration is seamless via JUnit XML reporting.
- **Negative**: Maintaining high-quality Evaluation Datasets (TestCase ground truths) requires significant ongoing engineering effort.
