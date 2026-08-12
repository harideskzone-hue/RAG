from app.agents.registry import AgentRegistry
from app.domain.models import AgentManifest

class AgentDependencyRegistry:
    """
    Formalizes the Agent Dependency Graph.
    Instead of LLMs guessing dependencies or arbitrary hardcoding, this tracks true domain dependencies.
    """
    def __init__(self, agent_registry: AgentRegistry | None = None):
        self.registry = agent_registry or AgentRegistry()
        
        # Hardcoded domain dependencies for MVP.
        # Can be extended to read from manifests in the future.
        self._dependencies = {
            "metadata_agent": [],
            "vector_agent": ["metadata_agent"],
            "video_agent": ["vector_agent"],
            "reasoning_agent": ["metadata_agent", "vector_agent", "video_agent"]
        }
        
    def get_dependencies(self, agent_name: str) -> list[str]:
        return self._dependencies.get(agent_name, [])
