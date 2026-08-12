# ADR 0008: Guardrails

## Problem
The Reasoning Engine and Final Response generator are LLMs. They are susceptible to hallucination, prompt injection, and outputting sensitive PII. An unconstrained output poses a massive risk to the VISTA platform's users.

## Decision
We implemented a strict, deterministic `GuardrailLayer` that executes immediately before presenting any response to the user. Governed by a `GuardrailManifest`, it runs discrete verifiers (PII, Jailbreak, Evidence, Confidence). A central `RiskAssessor` computes a final `RiskLevel`, and the orchestrator definitively blocks anything classified as HIGH or CRITICAL.

## Alternatives Considered
- System prompt instructions (e.g. "Do not hallucinate") (proven to be easily bypassed and ignored by LLMs).
- Post-processing regex only (insufficient for complex hallucinations or confidence thresholds).

## Consequences
- **Positive**: Guarantees output safety and protects user privacy. Nothing reaches the user without passing the gate.
- **Negative**: Adds latency to the final response. Overly aggressive guardrails may result in high false-positive rejection rates (blocked safe responses).
