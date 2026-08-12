from app.domain.memory.events import MemoryEventBus, MemoryCreated
from app.domain.memory.update_policy import MemoryUpdatePolicy
from app.domain.memory.repository import MemoryRepository
from app.domain.memory.retriever import MemoryRetriever
from app.domain.memory.ranker import MemoryRanker
from app.domain.memory.injector import MemoryInjector
from app.domain.memory.profile import MemoryProfile
from app.domain.memory.base import BaseMemory

class MemoryManager:
    """
    Orchestrator of the Memory lifecycle:
    Graph Event -> Update Policy -> Repositories -> Retriever -> Ranker -> Injector -> Agent
    """
    def __init__(self, repositories: dict[str, MemoryRepository], event_bus: MemoryEventBus):
        self.repositories = repositories
        self.event_bus = event_bus
        self.retriever = MemoryRetriever(repositories)
        self.ranker = MemoryRanker()
        self.injector = MemoryInjector()
        self.update_policy = MemoryUpdatePolicy()
        
    def add_memory(self, memory: BaseMemory):
        repo_type = memory.memory_type
        if repo_type in self.repositories:
            self.repositories[repo_type].save(memory.memory_id, memory)
            self.event_bus.publish(MemoryCreated(payload={"memory_id": str(memory.memory_id), "type": repo_type}))
            
    def get_memories_for_agent(self, agent_name: str, profile: MemoryProfile, query_params: dict) -> list[BaseMemory]:
        """Retrieves, ranks, and injects appropriate memory types for specific agents."""
        raw_memories = self.retriever.retrieve(profile, query_params)
        ranked_memories = self.ranker.rank(raw_memories)
        
        if agent_name == "planner":
            return self.injector.inject_for_planner(ranked_memories)
        elif agent_name == "reasoning":
            return self.injector.inject_for_reasoning(ranked_memories)
        elif agent_name == "report":
            return self.injector.inject_for_report(ranked_memories)
            
        return ranked_memories
