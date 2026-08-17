from typing import Dict, Any, List

class PlannerNode:
    """
    Deterministically decides which tools to execute based strictly on the QueryIntent.
    Does NOT infer meaning from the raw query or use the LLM.
    """
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if "abstain_reason" in state and state["abstain_reason"]:
            # Skip planning if intent parsing failed
            state["execution_plan"] = []
            return state
            
        intent = state.get("query_intent")
        if not intent:
            state["abstain_reason"] = "Missing structured intent."
            state["execution_plan"] = []
            return state
            
        plan = []
        
        # Deterministic logic based on LLM's structured semantic fields
        if intent.intent_type.value == "SEARCH":
            if intent.identity_target:
                plan.append("PersonSearchTool")
            else:
                plan.append("EvidenceSearchTool")
                
        elif intent.intent_type.value == "TIMELINE":
            plan.append("TimelineTool")
            
        elif intent.intent_type.value == "COUNT":
            plan.append("EvidenceSearchTool")
            plan.append("AggregationTool")
            
        elif intent.intent_type.value == "RELATIONSHIP":
            plan.append("PersonSearchTool")
            plan.append("TimelineTool")
            
        else:
            plan.append("EvidenceSearchTool") # fallback based on structured intent
            
        state["execution_plan"] = plan
        return state
