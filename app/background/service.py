import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTaskService:
    """
    Abstract representation of a background job queue.
    Currently uses asyncio.create_task for local execution.
    Can be swapped out for Celery/Redis RQ later without changing the API layer.
    """
    
    async def submit(self, job_id: str, func: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs):
        """
        Submits an asynchronous task to run in the background.
        """
        asyncio.create_task(self._run_and_log(job_id, func, *args, **kwargs))
        
    async def _run_and_log(self, job_id: str, func: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs):
        try:
            logger.info(f"Background job '{job_id}' started.")
            await func(*args, **kwargs)
            logger.info(f"Background job '{job_id}' completed successfully.")
        except Exception as e:
            logger.error(
                f"Background job '{job_id}' failed: {e}",
                exc_info=True
            )

