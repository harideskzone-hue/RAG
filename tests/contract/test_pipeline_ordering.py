from app.graph.validators.workflow_validator import WorkflowValidator
from app.schemas.context import VistaContext, UserContext, ExecutionPlan, ExecutionGroup

def test_pipeline_ordering_rejects_evidence_after_verification():
    validator = WorkflowValidator()
    
    # Create an invalid plan where video_agent runs after verification_agent
    plan = ExecutionPlan(
        success=True,
        agents=["metadata_agent", "verification_agent", "video_agent", "reasoning_agent"],
        execution_groups=[
            ExecutionGroup(agents=["metadata_agent"]),
            ExecutionGroup(agents=["verification_agent"]),
            ExecutionGroup(agents=["video_agent"]), # VIOLATION!
            ExecutionGroup(agents=["reasoning_agent"])
        ]
    )
    
    context = VistaContext(
        user=UserContext(user_id="1", role="admin"),
        execution_plan=plan
    )
    
    result = validator.validate(context)
    
    assert result.valid is False
    assert any("Pipeline ordering contract violated" in err for err in result.errors)
    assert any("Evidence producer 'video_agent'" in err for err in result.errors)

def test_pipeline_ordering_accepts_valid_sequence():
    validator = WorkflowValidator()
    
    # Create a valid plan
    plan = ExecutionPlan(
        success=True,
        agents=["metadata_agent", "video_agent", "evidence_fusion_agent", "verification_agent", "reasoning_agent"],
        execution_groups=[
            ExecutionGroup(agents=["metadata_agent", "video_agent"]),
            ExecutionGroup(agents=["evidence_fusion_agent"]),
            ExecutionGroup(agents=["verification_agent"]),
            ExecutionGroup(agents=["reasoning_agent"])
        ]
    )
    
    context = VistaContext(
        user=UserContext(user_id="1", role="admin"),
        execution_plan=plan
    )
    
    result = validator.validate(context)
    
    # We only care that it doesn't fail on pipeline ordering
    # It might fail on other things (e.g. tools, confidence) if we didn't mock them,
    # but let's check specifically that ordering errors are absent.
    ordering_errors = [err for err in result.errors if "Pipeline ordering contract violated" in err]
    assert len(ordering_errors) == 0
