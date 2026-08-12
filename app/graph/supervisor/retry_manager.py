import logging
import random


class RetryManager:
    """
    Manages retry policies for agents and tools.
    Policy: Retry → Fallback → Skip → Abort
    Uses exponential backoff with jitter to prevent thundering herd.
    """
    def __init__(self, max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 10.0):
        self.retry_counts: dict[str, int] = {}
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def should_retry(self, agent_name: str, error: Exception) -> bool:
        current = self.retry_counts.get(agent_name, 0)
        if current < self.max_retries:
            self.retry_counts[agent_name] = current + 1
            logging.info(f"Retrying agent {agent_name}. Attempt {current + 1}/{self.max_retries}")
            return True
        logging.warning(f"Agent {agent_name} exhausted all {self.max_retries} retries.")
        return False
    
    def get_backoff_delay(self, agent_name: str) -> float:
        """
        Calculate exponential backoff with jitter.
        Delay = min(base_delay * 2^attempt + jitter, max_delay)
        """
        attempt = self.retry_counts.get(agent_name, 1)
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        # Add ±25% jitter to prevent thundering herd
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)
        
    def reset(self):
        self.retry_counts.clear()

