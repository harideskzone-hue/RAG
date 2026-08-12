import json
import logging
from typing import Any

from app.agents.intent.schemas import IntentResult

logger = logging.getLogger(__name__)

EXPANDER_SYSTEM_PROMPT = """You are a strictly structured query expansion module for a CCTV video retrieval system.
Your job is to take the extracted entity and attributes and generate 2 to 4 semantic variations of the search query for vector retrieval.

CRITICAL RULE: DO NOT INVENT OR HALLUCINATE NEW FACTS. 
If the user specifies "red shirt", DO NOT add "male", "backpack", or "bicycle" unless explicitly provided in the input.
Only rewrite the provided facts using synonymous phrasing that might better match embedding spaces.

Format your output as a JSON object:
{
    "queries": [
        "query 1",
        "query 2",
        "query 3"
    ]
}
"""

EXPANDER_USER_PROMPT = """Entity: {entity}
Attributes: {attributes}
Original Description: {description}

Generate semantic variations based ONLY on these facts.
"""

class QueryExpander:
    def __init__(self, llm_client=None):
        self.llm = llm_client
        
    async def expand(self, intent_result: IntentResult) -> list[str]:
        entities = intent_result.entities or {}
        description = entities.get("description", "")
        
        # Fallback to single query if no LLM or empty description
        if not self.llm or not description:
            return [description] if description else []
            
        try:
            response = await self.llm.ainvoke([
                {"role": "system", "content": EXPANDER_SYSTEM_PROMPT},
                {"role": "user", "content": EXPANDER_USER_PROMPT.format(
                    entity=intent_result.intent.value,
                    attributes=json.dumps(entities),
                    description=description
                )}
            ])
            
            import re
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            parsed = json.loads(content)
            
            queries = parsed.get("queries", [])
            # Always ensure original description is in the pool
            if description not in queries:
                queries.insert(0, description)
            
            # Limit to 4 queries to prevent excessive load
            return queries[:4]
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Falling back to single query.")
            return [description]
