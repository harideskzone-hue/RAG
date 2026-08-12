from typing import Any

from app.memory.metrics import MemoryMetrics
from app.memory.policy import MemoryPolicy
from app.schemas.context import VistaContext


class ConversationSummarizer:
    """
    Summarizes conversation history when thresholds are exceeded.
    Does NOT summarize structured evidence.
    """
    def __init__(self, policy: MemoryPolicy, metrics: MemoryMetrics):
        self.policy = policy
        self.metrics = metrics

    def _estimate_tokens(self, messages: list[Any]) -> int:
        """
        Rough heuristic for token count. 
        In production, use tiktoken or similar.
        """
        tokens = 0
        for msg in messages:
            content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg.get("content", ""))
            tokens += len(content.split()) * 1.3 # Rough approximation
        return int(tokens)

    def execute(self, context: VistaContext):
        """
        Compresses conversation history in place if thresholds are exceeded.
        """
        if not self.policy.enable_summarization:
            return

        if not context.messages:
            return

        token_count = self._estimate_tokens(context.messages)
        message_count = len(context.messages)

        if token_count > self.policy.summary_threshold_tokens or message_count > self.policy.max_messages:
            # We would invoke an LLM here to summarize the oldest N messages.
            # For now, we mock the summarization behavior.
            
            # Keep the last 5 messages, summarize the rest
            keep_count = min(5, len(context.messages))
            to_summarize = context.messages[:-keep_count]
            retained = context.messages[-keep_count:]
            
            tokens_to_remove = self._estimate_tokens(to_summarize)
            
            summary_msg = {
                "role": "system", 
                "content": f"Summary of {len(to_summarize)} previous messages: [The user asked several queries about camera statuses and person searches. Investigations were conducted and reports generated.]"
            }
            
            context.messages = [summary_msg] + retained
            
            self.metrics.increment_summaries(1)
            self.metrics.increment_tokens_removed(tokens_to_remove)
