from app.agents.base_agent import BaseAgent
from app.agents.metadata.agent import MetadataAgent
from app.agents.vector.agent import VectorAgent
from app.agents.video.agent import VideoAgent
from app.agents.event.agent import EventAgent
from app.agents.reasoning.agent import ReasoningAgent
from app.agents.evidence.agent import EvidenceAgent
from app.agents.knowledge_graph.agent import KnowledgeGraphAgent
from app.graph.supervisor.event_bus import EventBus

AGENT_METADATA = "metadata_agent"
AGENT_VECTOR = "vector_agent"
AGENT_VIDEO = "video_agent"
AGENT_EVENT = "event_agent"
AGENT_REASONING = "reasoning_agent"
AGENT_EVIDENCE = "evidence_agent"
AGENT_REPORT = "report_agent"
AGENT_KG = "knowledge_graph_agent"

class AgentRegistry:
    """
    Central registry for all specialized agents in the VISTA AI system.
    Makes adding new agents (like Vehicle or Face agent) straightforward.
    """
    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent with name {agent.name} is already registered.")
        self._agents[agent.name] = agent

    def register_default_agents(self, event_bus: EventBus):
        self.register(MetadataAgent(event_bus))
        self.register(VectorAgent(event_bus))
        self.register(VideoAgent(event_bus))
        self.register(EventAgent(event_bus))
        self.register(ReasoningAgent())
        self.register(EvidenceAgent(event_bus))
        self.register(KnowledgeGraphAgent())

    def get_agent(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def get_all_agents(self) -> dict[str, BaseAgent]:
        return self._agents.copy()

# Singleton registry
agent_registry = AgentRegistry()
