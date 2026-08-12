import asyncio
import logging
from collections.abc import Callable
from typing import Any


class TimeoutManager:
    """
    Enforces timeout limits for agents.
    """
    def __init__(self):
        self.default_timeout = 30 # seconds

    async def execute_with_timeout(self, coro: Callable, timeout_ms: int = None) -> Any:
        timeout_seconds = (timeout_ms / 1000.0) if timeout_ms else self.default_timeout
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logging.error(f"Execution timed out after {timeout_seconds}s")
            raise TimeoutError(f"Execution exceeded timeout of {timeout_seconds}s")
