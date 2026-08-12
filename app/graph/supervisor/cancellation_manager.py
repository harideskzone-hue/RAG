import asyncio
import logging

from app.graph.supervisor.state_machine import ExecutionState, StateMachine


class CancellationManager:
    """
    Manages workflow cancellation (e.g. user closed browser).
    Ensures expensive calls (VLM) are aborted and resources released.
    """
    def __init__(self, state_machine: StateMachine):
        self.state_machine = state_machine
        self.cancel_event = asyncio.Event()

    def cancel(self):
        logging.info("Cancellation requested.")
        self.cancel_event.set()
        # Transition state
        if self.state_machine.state not in [ExecutionState.COMPLETED, ExecutionState.FAILED]:
            self.state_machine.transition_to(ExecutionState.CANCELLED)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()
