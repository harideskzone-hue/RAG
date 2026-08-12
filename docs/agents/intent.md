# Intent Agent

## Purpose
Classifies incoming user queries to determine the necessary downstream agents required.

## Inputs
- `query` (str): Raw user query
- `conversation_history` (list): Past messages for context

## Outputs
- `IntentResult`: Contains the primary intent classification (e.g., `STATUS_CHECK`, `PERSON_SEARCH`, `VIDEO_ANALYSIS`) and extracted entities.

## Dependencies
- Requires LLM Adapter for zero-shot classification.

## Failure Behavior
If intent cannot be determined, defaults to a `FALLBACK` intent which triggers the Supervisor to ask the user for clarification.
