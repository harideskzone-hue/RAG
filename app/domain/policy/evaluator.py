from app.domain.policy.context import PolicyContext
from app.domain.policy.rules import PolicyRule, RuleResult
from typing import Any

class RuleEvaluator:
    """Matches the PolicyContext and ExecutionPlan against all active rules."""
    
    @staticmethod
    def evaluate(rules: list[PolicyRule], context: PolicyContext, plan: Any) -> list[RuleResult]:
        results = []
        # Sort rules by priority descending
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            result = rule.evaluate(context, plan)
            if result.matched:
                results.append(result)
                
        return results
