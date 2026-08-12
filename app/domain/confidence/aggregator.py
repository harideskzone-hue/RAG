from app.schemas.context import VistaContext
from app.agents.registry import AgentRegistry, agent_registry

class ConfidenceAggregator:
    """
    Deterministically aggregates agent confidences using a weighted average.
    Weights are derived from the agent's baseline reliability defined in their manifest.
    """
    def __init__(self, registry: AgentRegistry | None = None):
        self.registry = registry or agent_registry
        
    def aggregate(self, context: VistaContext) -> float:
        if not context.results:
            return 0.0
            
        total_weighted_confidence = 0.0
        total_reliability = 0.0
        
        for agent_name, result in context.results.items():
            if result.status == "SUCCESS":
                agent = self.registry.get_agent(agent_name)
                # If no manifest, default reliability to 0.5
                reliability = agent.manifest.reliability if agent and hasattr(agent, 'manifest') and agent.manifest else 0.5

                
                agent_conf = result.confidence
                total_weighted_confidence += (agent_conf * reliability)
                total_reliability += reliability
                
        if total_reliability == 0.0:
            return 0.0
            
        overall_confidence = total_weighted_confidence / total_reliability
        return overall_confidence
