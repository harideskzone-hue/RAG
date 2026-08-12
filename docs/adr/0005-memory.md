# ADR 0005: Investigation Memory

## Problem
The system needs to retain context across massive, multi-day investigations. Standard conversational memory (chat history) is unstructured and quickly overflows the LLM context window, causing catastrophic forgetting of critical evidence.

## Decision
We built a structured `InvestigationMemory` layer split by episodic and semantic boundaries. The Memory acts as a repository distinct from the Knowledge Graph. While the Graph holds universal entity facts, the Memory holds the specific narrative, hypothesis trajectory, and context of the *current investigation*. 

## Alternatives Considered
- **Sliding Window Chat History**: (Inadequate for retaining older, but highly relevant, forensic facts).
- **Injecting the entire Knowledge Graph**: (Too massive; exceeds context limits).

## Consequences
- **Positive**: The Reasoning Engine has precise, context-bounded memory. We can utilize RAG (Retrieval-Augmented Generation) directly on past reasoning traces.
- **Negative**: We must manage complex caching, retrieval, and ranking mechanisms in the Memory domain to ensure the LLM receives the right snippets.
