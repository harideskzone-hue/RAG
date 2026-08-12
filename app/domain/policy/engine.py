from app.domain.policy.context import PolicyContext
from app.domain.policy.repository import PolicyRepository
from app.domain.policy.evaluator import RuleEvaluator
from app.domain.policy.estimator import CostEstimator
from app.domain.policy.resolver import ConflictResolver
from app.domain.policy.decision import PolicyDecision
from app.domain.policy.explanation import PolicyExplanation
from app.domain.policy.trace import PolicyTrace
from app.domain.policy.statistics import PolicyStatistics
from uuid import uuid4
from typing import Any

class PolicyEngine:
    """
    The central orchestrator: 
    PolicyContext -> Repository -> Evaluator -> Estimator -> Resolver -> Decision -> Explanation -> Trace -> Statistics
    """
    def __init__(self, repository: PolicyRepository, policy_id: str = "default_investigation"):
        self.repository = repository
        self.policy_id = policy_id
        self.statistics = PolicyStatistics()
        
    def evaluate_plan(self, context: PolicyContext, plan: Any) -> tuple[PolicyExplanation, PolicyTrace]:
        self.statistics.policies_executed += 1
        trace = PolicyTrace(execution_id=str(uuid4()))
        
        # 1. Load Rules
        rules = self.repository.load_rules(self.policy_id)
        
        # 2. Estimate Costs
        estimated_costs = CostEstimator.estimate(plan)
        
        # 3. Evaluate Rules
        results = RuleEvaluator.evaluate(rules, context, plan)
        
        if results:
            self.statistics.policies_matched += 1
            for res in results:
                # Assuming RuleResult doesn't carry rule_id right now in MVP, we log action
                trace.add_event(rule_id="evaluator", action_taken=res.recommended_action, details=res.reason)
        
        # 4. Resolve Conflicts
        decision, reason, affected_agents, modified_plan, primary_rule_id = ConflictResolver.resolve(results, plan)
        
        # 5. Build Explanation
        explanation = PolicyExplanation(
            decision=decision,
            primary_rule_id=primary_rule_id,
            reason=reason,
            affected_agents=affected_agents,
            estimated_cost=estimated_costs.cost_usd,
            estimated_latency=estimated_costs.latency_ms,
            original_plan=plan,
            validated_plan=modified_plan
        )
        
        # 6. Update Statistics & Trace
        self._update_statistics_and_trace(decision, reason, estimated_costs, modified_plan, trace)
            
        return explanation, trace

    def _update_statistics_and_trace(self, decision, reason, estimated_costs, modified_plan, trace):
        if decision == PolicyDecision.REJECT:
            self.statistics.rejected_plans += 1
            self.statistics.execution_savings_usd += estimated_costs.cost_usd
        elif decision == PolicyDecision.MODIFY:
            self.statistics.modified_plans += 1
            modified_costs = CostEstimator.estimate(modified_plan)
            self.statistics.execution_savings_usd += (estimated_costs.cost_usd - modified_costs.cost_usd)
        elif decision == PolicyDecision.DEFER:
            self.statistics.deferred_plans += 1
        else:
            self.statistics.approved_plans += 1
            
        trace.add_event(
            rule_id="resolver",
            action_taken="FINAL_DECISION",
            details=f"Decision: {decision.value}. Reason: {reason}"
        )
