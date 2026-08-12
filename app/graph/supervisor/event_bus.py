import logging
import threading
from collections import deque
from collections.abc import Callable

from app.graph.supervisor.telemetry import AgentEvent

logger = logging.getLogger(__name__)


class EventBus:
    """
    Decouples agents by using an event-driven architecture.
    Agents publish events (e.g. METADATA_COMPLETE) which are routed here.
    
    Thread-safe: uses a lock to protect subscribers and history
    from concurrent modification across requests.
    """
    MAX_HISTORY = 10_000

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Callable[[AgentEvent], None]]] = {}
        self.history: deque[AgentEvent] = deque(maxlen=self.MAX_HISTORY)

    def subscribe(self, event_type: str, callback: Callable[[AgentEvent], None]):
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def publish(self, event: AgentEvent):
        with self._lock:
            self.history.append(event)
            callbacks = list(self._subscribers.get(event.event_type, []))
        # Execute callbacks outside the lock to prevent deadlocks
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"EventBus callback error for {event.event_type}: {e}")

    def unsubscribe(self, event_type: str, callback: Callable[[AgentEvent], None]):
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb is not callback
                ]

