from typing import Any

from app.memory.eviction import EvictionPolicy
from app.memory.metrics import MemoryMetrics
from app.memory.policy import MemoryPolicy
from app.memory.summarizer import ConversationSummarizer
from app.schemas.context import VistaContext


class MemoryManager:
    """
    Orchestrates Memory Optimization: Summarization and Eviction.
    Acts as an explicit node in the LangGraph workflow.
    """
    def __init__(self, policy: MemoryPolicy = None):
        self.policy = policy or MemoryPolicy()
        self.metrics = MemoryMetrics()
        self.eviction = EvictionPolicy(self.policy, self.metrics)
        self.summarizer = ConversationSummarizer(self.policy, self.metrics)

    async def run(self, context: VistaContext) -> dict[str, Any]:
        """
        Executes the memory optimization pipeline on the context.
        Returns a dict indicating state updates (if any).
        """
        # 1. Summarize conversation if necessary
        self.summarizer.execute(context)
        
        # 2. Evict stale evidence/data
        self.eviction.execute(context)
        
        # 3. Update memory size metric (mock calculation)
        self.metrics.memory_size_bytes = len(str(context.model_dump()))
        
        # Return state updates (for LangGraph state reduction if needed, though we mutated context in-place)
        return {"metrics": self.metrics.model_dump()}
