import json
from typing import Dict, Any

from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMRequest
from app.schemas.intent import QueryIntent

class IntentNode:
    """
    Parses natural language into a structured QueryIntent using ONLY the LLM.
    Strict architectural invariant: No keyword matching.
    """
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        
        self.system_prompt = """You are the Semantic Intent Router for VISTA AI.
Your ONLY job is to translate the user's natural language question into a structured QueryIntent JSON object.
You must not attempt to answer the question yourself or invent facts.

Extract:
- intent_type: SEARCH, COUNT, TIMELINE, RELATIONSHIP, UNKNOWN
- entity_type: PERSON, VEHICLE, OBJECT, UNKNOWN
- temporal_constraints: start_time_str, end_time_str, is_relative (e.g. "last 5 minutes")
- spatial_constraints: locations, camera_ids
- attributes: physical descriptions (e.g. {"clothing": "red shirt"})
- requested_evidence: what data the user wants (e.g. ["identity", "timeline", "counts"])
- confidence: Your confidence in parsing this query (0.0 to 1.0)
- is_valid: False if the query is a greeting, nonsense, or unparseable.
- clarification_needed: If ambiguous or invalid, explain what you need from the user.

Output strictly in JSON matching the schema."""

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        
        if not query:
            raise ValueError("Query cannot be empty")
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Parse this query into a QueryIntent: '{query}'"}
        ]
        
        request = LLMRequest(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # We enforce structural generation via domain client, zero Groq-specific imports.
        intent: QueryIntent = await self.llm.generate_structured(request, QueryIntent)
        
        if not intent.is_valid or intent.confidence < 0.6 or intent.clarification_needed:
            # ABSTAIN logic kicks in early if intent is unclear
            state["abstain_reason"] = intent.clarification_needed or "I could not confidently determine your intent."
            state["query_intent"] = None
        else:
            state["query_intent"] = intent
            
        return state
