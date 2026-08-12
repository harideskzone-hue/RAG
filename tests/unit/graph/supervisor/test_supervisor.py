
import pytest

from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.failure_handler import FailureHandler
from app.graph.supervisor.retry_manager import RetryManager
from app.graph.supervisor.state_machine import ExecutionState, StateMachine
from app.graph.supervisor.telemetry import AgentEvent
from app.schemas.context import VistaContext


def test_state_machine_valid_transitions():
    sm = StateMachine()
    assert sm.state == ExecutionState.CREATED
    sm.transition_to(ExecutionState.PLANNED)
    sm.transition_to(ExecutionState.VALIDATED)
    sm.transition_to(ExecutionState.RUNNING)
    sm.transition_to(ExecutionState.COMPLETED)
    assert sm.state == ExecutionState.COMPLETED

def test_state_machine_invalid_transition():
    sm = StateMachine()
    with pytest.raises(ValueError):
        sm.transition_to(ExecutionState.COMPLETED) # Cannot jump from CREATED to COMPLETED

def test_event_bus():
    bus = EventBus()
    received = []
    
    def callback(event: AgentEvent):
        received.append(event)
        
    bus.subscribe("TEST_EVENT", callback)
    
    event = AgentEvent(
        agent_name="test_agent",
        event_type="TEST_EVENT",
        start_time=0.0,
        status="RUNNING",
        trace_id="123"
    )
    
    bus.publish(event)
    
    assert len(received) == 1
    assert received[0].agent_name == "test_agent"
    
def test_retry_manager():
    rm = RetryManager()
    error = ValueError("Test error")
    
    assert rm.should_retry("metadata", error) == True
    assert rm.should_retry("metadata", error) == True
    assert rm.should_retry("metadata", error) == True
    assert rm.should_retry("metadata", error) == False # Max retries (3) reached
    
def test_failure_handler_graceful_degradation():
    fh = FailureHandler()
    sm = StateMachine()
    sm.transition_to(ExecutionState.PLANNED)
    sm.transition_to(ExecutionState.VALIDATED)
    sm.transition_to(ExecutionState.RUNNING)
    
    # Mock context
    from app.schemas.context import ExecutionPlan, UserContext
    context = VistaContext(user=UserContext(user_id="1", role="admin"), conversation_id="1")
    context.execution_plan = ExecutionPlan(success=True)
    
    # Video can be gracefully degraded
    can_continue = fh.handle_failure("video_agent", ValueError("Failed"), context, sm)
    assert can_continue == True
    assert sm.state == ExecutionState.RUNNING # Didn't fail entirely
    
    # Metadata failure is fatal
    context.execution_plan.agents = ["video_agent"]
    can_continue = fh.handle_failure("metadata_agent", ValueError("Failed"), context, sm)
    assert can_continue == False
    assert sm.state == ExecutionState.FAILED # Fatal failure
