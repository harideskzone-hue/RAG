from app.domain.policy.rules import RuleResult
from app.domain.policy.decision import PolicyDecision
from typing import Any

class ConflictResolver:
    """Determines the outcome when conflicting rules fire."""
    
    @staticmethod
    def resolve(results: list[RuleResult], original_plan: Any) -> tuple[PolicyDecision, str, list[str], Any, str]:
        """
        Returns: (Decision, Reason, Affected Agents, Modified Plan, Primary Rule ID)
        """
        if not results:
            return PolicyDecision.APPROVE, "No rules matched", [], original_plan, ""
            
        # Evaluator passes results in descending priority order. 
        # The first matched result dictates the primary action.
        highest_priority_result = results[0]
        decision = PolicyDecision(highest_priority_result.recommended_action)
        reason = highest_priority_result.reason
        affected_agents = highest_priority_result.affected_agents
        
        # Simple simulated modification for the MVP
        modified_plan = original_plan
        if decision == PolicyDecision.MODIFY and isinstance(original_plan, list):
            modified_plan = [agent for agent in original_plan if agent not in affected_agents]
            
        return decision, reason, affected_agents, modified_plan, "highest_priority_rule"
