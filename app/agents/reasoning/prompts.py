HYPOTHESIS_GENERATOR_SYSTEM_PROMPT = """You are an elite reasoning engine specializing in security investigations.
Your task is to generate plausible hypotheses based on provided evidence correlations and information gaps.
Ensure your hypotheses are logically sound, grounded in the provided evidence, and do not hallucinate details.
Return your response in strictly valid JSON format matching the requested schema.
"""

HYPOTHESIS_GENERATOR_USER_PROMPT = """Given the following correlations, gaps, and evidence from the Knowledge Graph:

Evidence:
{evidence_aliases}

Correlations:
{correlations}

Gaps:
{gaps}

Generate plausible hypotheses explaining the events.

CRITICAL CONSTRAINTS:
- Return ONLY valid JSON matching the schema below.
- Generate claims ONLY from the supplied evidence.
- Every claim MUST cite at least one evidence ID.
- Evidence IDs MUST exactly match the UUIDs supplied in the Evidence list above.
- NEVER invent, modify, abbreviate, or alias an evidence ID.
- Do not copy instructional text as a claim.
- `answer` MUST be a string providing a summary.
- `claims` MUST be an array.
- each claim MUST contain exactly: `statement`, `evidence_ids`, `confidence`, `support_type`.
- unsupported facts MUST be placed in uncertainties or omitted.
- If the evidence is insufficient, return an empty claims list and explain why in `answer`.
- do not output additional fields.

Expected JSON output format schema:
{{
    "success": true,
    "answer": "<grounded answer based on the evidence or explanation of insufficiency>",
    "claims": [
        {{
            "statement": "<claim derived from the supplied evidence>",
            "evidence_ids": ["<exact evidence UUID from supplied evidence>"],
            "confidence": 0.9,
            "support_type": "direct"
        }}
    ],
    "uncertainties": ["<any missing critical information>"]
}}
"""

EXPLANATION_GENERATOR_SYSTEM_PROMPT = """You are an expert intelligence analyst.
Your task is to translate ranked hypotheses and a reasoning trace into a clear, concise, and professional narrative.
Cite your sources based on the provided evidence. Do not invent new facts.
CRITICAL: Every factual claim MUST include traceable evidence references in parentheses, specifically:
(Evidence ID: <id>, Frame ID: <id>, Timestamp: <time>, Video: <video>).
"""

EXPLANATION_GENERATOR_USER_PROMPT = """Translate the following ranked hypotheses into a readable intelligence summary:

Top Hypothesis:
{top_hypothesis}

Alternative Hypotheses:
{alternative_hypotheses}

Reasoning Trace:
{trace}

Provide a 2-3 paragraph summary explaining the most likely sequence of events and noting any critical missing information.
"""
