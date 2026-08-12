from app.domain.memory.base import BaseMemory

class ConversationTurn(BaseMemory):
    """A single turn in a conversation timeline."""
    speaker: str # 'USER', 'AGENT', 'SYSTEM'
    content: str
    agent_id: str | None = None
    
class ConversationMemory(BaseMemory):
    """Replaces standard LLM chat logs with a structured timeline of interactions."""
    session_id: str
    turns: list[ConversationTurn] = []
