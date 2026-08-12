from enum import Enum, auto


class ExecutionState(Enum):
    CREATED = auto()
    PLANNED = auto()
    VALIDATED = auto()
    RUNNING = auto()
    WAITING = auto()
    RETRYING = auto()
    FAILED = auto()
    COMPLETED = auto()
    CANCELLED = auto()

class StateMachine:
    """
    Manages the lifecycle of a workflow execution.
    Prevents invalid transitions.
    """
    def __init__(self):
        self._state = ExecutionState.CREATED
        self._valid_transitions = {
            ExecutionState.CREATED: [ExecutionState.PLANNED, ExecutionState.RUNNING, ExecutionState.CANCELLED],
            ExecutionState.PLANNED: [ExecutionState.VALIDATED, ExecutionState.FAILED, ExecutionState.CANCELLED],
            ExecutionState.VALIDATED: [ExecutionState.RUNNING, ExecutionState.FAILED, ExecutionState.CANCELLED],
            ExecutionState.RUNNING: [ExecutionState.WAITING, ExecutionState.COMPLETED, ExecutionState.FAILED, ExecutionState.CANCELLED, ExecutionState.RETRYING],
            ExecutionState.WAITING: [ExecutionState.RUNNING, ExecutionState.CANCELLED],
            ExecutionState.RETRYING: [ExecutionState.RUNNING, ExecutionState.FAILED, ExecutionState.CANCELLED],
            ExecutionState.FAILED: [ExecutionState.RETRYING, ExecutionState.COMPLETED], # Completed if gracefully degraded
            ExecutionState.COMPLETED: [],
            ExecutionState.CANCELLED: []
        }

    @property
    def state(self) -> ExecutionState:
        return self._state

    def transition_to(self, new_state: ExecutionState) -> bool:
        if new_state in self._valid_transitions[self._state]:
            self._state = new_state
            return True
        raise ValueError(f"Invalid state transition from {self._state.name} to {new_state.name}")
