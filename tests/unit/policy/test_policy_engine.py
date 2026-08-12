import pytest
from app.domain.policy.engine import PolicyEngine
from app.domain.policy.context import PolicyContext
from app.domain.policy.budget import ExecutionBudget
from app.domain.policy.config import get_default_repository
from app.domain.policy.decision import PolicyDecision

def test_policy_engine_lifecycle():
    # 1. Setup
    repo = get_default_repository()
    engine = PolicyEngine(repository=repo)
    
    # 2. Context
    budget = ExecutionBudget(max_cost_usd=0.5)
    context = PolicyContext(
        execution_mode="INVESTIGATION",
        memory_profile="INVESTIGATION",
        budget=budget,
        confidence_threshold=0.8
    )
    
    # 3. Execution Plan (Mock)
    plan = ["metadata_agent", "video_agent"]
    
    # 4. Evaluate
    explanation, trace = engine.evaluate_plan(context, plan)
    
    # 5. Assertions
    # Our default rules are placeholders that return matched=False right now.
    # So the ConflictResolver will hit the "No rules matched" path, returning APPROVE
    assert explanation.decision == PolicyDecision.APPROVE
    assert len(trace.events) == 1
    assert trace.events[0].action_taken == "FINAL_DECISION"
    assert engine.statistics.policies_executed == 1
    assert engine.statistics.approved_plans == 1
