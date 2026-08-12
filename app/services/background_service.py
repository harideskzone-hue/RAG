import asyncio
import logging

class BackgroundTaskService:
    """
    Manages background tasks (fire-and-forget) to ensure they are tracked and errors are logged.
    Prevents silent failures and allows graceful shutdown.
    """
    def __init__(self):
        self.background_tasks = set()

    def create_task(self, coro, name: str = None) -> asyncio.Task:
        """
        Create a tracked background task.
        """
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.add(task)
        
        # Add a callback to remove the task from the set when it's done
        # and log any unhandled exceptions
        task.add_done_callback(self._handle_task_result)
        return task
        
    def _handle_task_result(self, task: asyncio.Task):
        self.background_tasks.discard(task)
        try:
            # We must retrieve the result to avoid "Task exception was never retrieved"
            # However, if it was cancelled, we ignore that
            if not task.cancelled():
                exc = task.exception()
                if exc:
                    logging.error(f"Background task {task.get_name()} failed with unhandled exception: {exc}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"Error while retrieving result for background task {task.get_name()}: {e}")

    async def shutdown(self, timeout: float = 5.0):
        """
        Cancel all pending background tasks and wait for them to finish.
        """
        if not self.background_tasks:
            return
            
        logging.info(f"Shutting down {len(self.background_tasks)} background tasks.")
        for task in self.background_tasks:
            task.cancel()
            
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
