from app.domain.memory.repository import MemoryRepository
from app.domain.memory.profile import MemoryProfile
from app.domain.memory.base import BaseMemory

class MemoryRetriever:
    """Strictly separates retrieval from management. Modifies queries based on MemoryProfile."""
    def __init__(self, repositories: dict[str, MemoryRepository]):
        self.repositories = repositories
        
    def retrieve(self, profile: MemoryProfile, query_params: dict) -> list[BaseMemory]:
        memories = []
        allowed_types = self._get_allowed_types(profile)
        
        for repo_type, repo in self.repositories.items():
            if repo_type in allowed_types:
                # In production, we pass query_params to a Vector DB or SQL DB here
                memories.extend(repo.search(**query_params))
                
        return memories
        
    def _get_allowed_types(self, profile: MemoryProfile) -> list[str]:
        if profile == MemoryProfile.SIMPLE:
            return ["ConversationMemory", "SummaryMemory"]
        elif profile == MemoryProfile.ITERATIVE:
            return ["ConversationMemory", "SummaryMemory", "EntityMemory"]
        elif profile == MemoryProfile.INVESTIGATION:
            return ["ConversationMemory", "SummaryMemory", "EntityMemory", "EpisodeMemory", "FacilityMemory"]
        # DEEP_INVESTIGATION
        return list(self.repositories.keys())
