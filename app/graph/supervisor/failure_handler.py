from enum import Enum
import logging

from app.graph.supervisor.state_machine import ExecutionState, StateMachine
from app.schemas.context import VistaContext

class RecoveryAction(str, Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    REPLAN = "replan"

class FailureHandler:
    """
    Decides how to handle failures that exhaust retries.
    Strict recovery policies: retry -> fallback -> skip -> abort -> replan.
    """
    def handle_failure(self, agent_name: str, error: Exception, context: VistaContext, state_machine: StateMachine) -> bool:
        """
        Returns True if execution can continue (skip/fallback), False if aborting.
        """
        logging.error(f"Handling ultimate failure for agent {agent_name}: {error}")
        
        policies = {
            "metadata_agent": RecoveryAction.RETRY,
            "vector_agent": RecoveryAction.RETRY,
            "video_agent": RecoveryAction.SKIP,
            "event_agent": RecoveryAction.SKIP,
            "report_agent": RecoveryAction.SKIP,
            "reasoning_agent": RecoveryAction.ABORT,
            "guardrail_agent": RecoveryAction.ABORT,
            "evidence_agent": RecoveryAction.ABORT,
            "knowledge_graph_agent": RecoveryAction.ABORT,
        }
        action = policies.get(agent_name, RecoveryAction.ABORT)
            
        logging.info(f"Recovery policy for {agent_name} failure: {action.value}")
        
        if action == RecoveryAction.SKIP:
            return True
            
        if action in [RecoveryAction.ABORT, RecoveryAction.RETRY]:
            state_machine.transition_to(ExecutionState.FAILED)
            return False
            
        return False
