from typing import Dict, Any

from app.domain.llm.base import BaseLLMClient
from app.domain.llm.models import LLMRequest

class ResponseNode:
    """
    Formulates a natural language response strictly based on the VerifiedResultContract.
    Does not invent facts or bypass the contract.
    """
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        
        self.system_prompt = """You are the VISTA AI Response Generator.
Your ONLY job is to explain the verified evidence to the user in natural language.
You MUST NOT invent, add, or alter any evidence.
If you are given an 'ABSTAIN' signal, politely inform the user that you cannot answer the question based on the evidence.
"""

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        abstain_reason = state.get("abstain_reason")
        contract = state.get("verified_contract")
        
        if abstain_reason:
            # Explicit Abstention Path
            prompt = f"The system had to ABSTAIN from answering the query: '{query}'.\nReason: {abstain_reason}\nProvide a polite, safe response to the user."
        elif contract:
            prompt = f"User query: '{query}'\n\nVerified Evidence:\n{contract.model_dump_json(indent=2)}\n\nProvide a natural language summary of this evidence."
        else:
            prompt = "No evidence or abstain reason found. Internal error."
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        request = LLMRequest(messages=messages)
        
        # Generation via abstract client
        response = await self.llm.generate(request)
        
        state["final_response"] = response.content
        return state
