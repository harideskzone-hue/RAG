from app.domain.memory.base import BaseMemory
from app.domain.memory.conversation import ConversationMemory
from app.domain.memory.summary import SummaryMemory
from app.domain.memory.entity import EntityMemory
from app.domain.memory.episode import EpisodeMemory
from app.domain.memory.facility import FacilityMemory

class MemoryInjector:
    """Dynamically decides which pieces of ranked memory go to which agents."""
    
    @staticmethod
    def inject_for_planner(memories: list[BaseMemory]) -> list[BaseMemory]:
        """Planner only needs Conversation to understand user intent."""
        return [m for m in memories if isinstance(m, ConversationMemory)]
        
    @staticmethod
    def inject_for_reasoning(memories: list[BaseMemory]) -> list[BaseMemory]:
        """Reasoning Engine needs domain facts."""
        return [m for m in memories if isinstance(m, (EntityMemory, FacilityMemory, EpisodeMemory, SummaryMemory))]
        
    @staticmethod
    def inject_for_report(memories: list[BaseMemory]) -> list[BaseMemory]:
        """Report Generator needs high-level summaries and conversation history."""
        return [m for m in memories if isinstance(m, (SummaryMemory, ConversationMemory))]
