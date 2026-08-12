import hashlib

from app.schemas.context import ExecutionPlan


class PlannerCache:
    """
    Caches identical execution plans to avoid redundant LLM invocations and latency.
    """
    
    def __init__(self):
        # In production, this would be backed by Redis.
        # For now, it's an in-memory dictionary.
        self._cache = {}

    def _generate_key(self, query: str, intent: str) -> str:
        """
        Generates a deterministic hash key for the query and intent.
        """
        key_str = f"{query.strip().lower()}|{intent}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def get(self, query: str, intent: str) -> ExecutionPlan | None:
        """
        Retrieve a cached execution plan.
        """
        key = self._generate_key(query, intent)
        return self._cache.get(key)

    def set(self, query: str, intent: str, plan: ExecutionPlan) -> None:
        """
        Store an execution plan in the cache.
        """
        key = self._generate_key(query, intent)
        self._cache[key] = plan

# Singleton instance for now
planner_cache = PlannerCache()
