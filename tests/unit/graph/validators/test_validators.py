from app.schemas.context import ExecutionGroup
from app.graph.validators.workflow_validator import WorkflowValidator
from app.schemas.context import (
    ExecutionPlan,
    ToolRequirement,
    UserContext,
    VistaContext,
)


def get_base_context():
    return VistaContext(
        user=UserContext(user_id="123", role="admin"),
        conversation_id="c1",
        execution_plan=ExecutionPlan(
            success=True,
            agents=["metadata", "vector"],
            tools=[ToolRequirement(name="postgres")],
            execution_groups=[ExecutionGroup(agents=["metadata"]), ExecutionGroup(agents=["vector"])],
            dependencies={"vector": ["metadata"]},
        )
    )

def test_valid_plan():
    validator = WorkflowValidator()
    context = get_base_context()
    
    result = validator.validate(context)
    assert result.valid is True
    assert len(result.errors) == 0

def test_duplicate_agents():
    validator = WorkflowValidator()
    context = get_base_context()
    context.execution_plan.agents.append("metadata")
    
    result = validator.validate(context)
    assert result.valid is False
    assert any("Duplicate agent" in err for err in result.errors)

def test_tool_dependency_missing():
    validator = WorkflowValidator()
    context = get_base_context()
    # Require milvus but vector agent is missing
    context.execution_plan.agents = ["metadata"]
    context.execution_plan.tools.append(ToolRequirement(name="milvus"))
    
    result = validator.validate(context)
    assert result.valid is False
    assert any("requires the 'vector' agent" in err for err in result.errors)

def test_circular_dependency():
    validator = WorkflowValidator()
    context = get_base_context()
    # metadata -> vector -> metadata
    context.execution_plan.dependencies = {
        "vector": ["metadata"],
        "metadata": ["vector"]
    }
    
    result = validator.validate(context)
    assert result.valid is False
    assert any("Circular dependency" in err for err in result.errors)

def test_execution_group_violation():
    validator = WorkflowValidator()
    context = get_base_context()
    # vector depends on metadata, but they are in the same group
    context.execution_plan.execution_groups = [["metadata", "vector"]]
    
    result = validator.validate(context)
    assert result.valid is False
    assert any("is in the same or a later execution group" in err for err in result.errors)

def test_permission_rejection():
    validator = WorkflowValidator()
    context = get_base_context()
    # User is operator, trying to use 'event' agent and 'delete_tool'
    context.user.role = "operator"
    context.execution_plan.agents.append("event")
    context.execution_plan.tools.append(ToolRequirement(name="delete_tool"))
    
    result = validator.validate(context)
    assert result.valid is False
    assert any("not allowed to use agent 'event'" in err for err in result.errors)
    assert any("not allowed to use tool 'delete_tool'" in err for err in result.errors)

def test_cost_latency_validation():
    validator = WorkflowValidator(max_token_limit=1000, max_latency_ms=100)
    context = get_base_context()
    context.execution_plan.estimated_tokens = 5000
    context.execution_plan.estimated_latency_ms = 500
    
    result = validator.validate(context)
    assert result.valid is False
    assert len(result.errors) == 2

def test_risk_approval():
    validator = WorkflowValidator()
    context = get_base_context()
    context.execution_plan.risk_level = "CRITICAL"
    
    result = validator.validate(context)
    assert result.valid is True # High risk doesn't invalidate, it requires approval
    assert result.approval_required is True
