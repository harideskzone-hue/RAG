from app.schemas.context import ExecutionPlan
from app.agents.registry import AgentRegistry

class CostOptimizer:
    """
    Optimizes ExecutionPlan for SIMPLE mode by pruning high-latency / high-cost agents.
    """
    def __init__(self, registry: AgentRegistry | None = None):
        self.registry = registry or AgentRegistry()
        
    def optimize(self, plan: ExecutionPlan) -> ExecutionPlan:
        # Simple policy: keep only agents with latency < 1000ms for SIMPLE queries
        # Or hardcode a simpler heuristic: drop VLM agents
        optimized_agents = []
        for agent_name in plan.agents:
            agent = self.registry.get_agent(agent_name)
            if agent:
                manifest = agent.manifest
                # If requires VLM, it's expensive. Skip it for SIMPLE mode.
                # Since VLM agents are expensive, we can check their capabilities or dependencies
                # ReasoningAgent uses VLM, VideoAgent might. 
                # Our AgentManifest doesn't have a direct `requires_vlm` property yet, 
                # but we can check if it depends on "gemini" or "vlm".
                if "gemini" not in manifest.dependencies and "vlm" not in manifest.dependencies:
                    optimized_agents.append(agent_name)
            else:
                optimized_agents.append(agent_name) # Keep if unknown
                
        # Ensure we always have at least one agent (e.g. metadata)
        if not optimized_agents and plan.agents:
            optimized_agents = [plan.agents[0]]
            
        plan.agents = optimized_agents
        
        # Prune execution_groups
        new_groups = []
        for group in plan.execution_groups:
            new_group = [a for a in group if a in optimized_agents]
            if new_group:
                new_groups.append(new_group)
        plan.execution_groups = new_groups
        
        # Prune tasks if they exist
        new_tasks = [t for t in plan.tasks if t.agent_type in optimized_agents]
        for t in new_tasks:
            t.dependencies = [dep for dep in t.dependencies if any(dep.endswith(f"_{a}") for a in optimized_agents)]
        plan.tasks = new_tasks
        
        # Prune tools if they exist
        if plan.tools:
            new_tools = []
            for tool in plan.tools:
                # Keep if any optimized agent requires it
                required = False
                for agent_name in optimized_agents:
                    agent = self.registry.get_agent(agent_name)
                    if agent and tool.name in agent.manifest.dependencies:
                        required = True
                        break
                if required:
                    new_tools.append(tool)
            plan.tools = new_tools
        
        # Update requires_vlm flag based on remaining agents
        vlm_required = False
        for agent_name in optimized_agents:
            agent = self.registry.get_agent(agent_name)
            if agent:
                if "gemini" in agent.manifest.dependencies or "vlm" in agent.manifest.dependencies:
                    vlm_required = True
        plan.requires_vlm = vlm_required
        
        return plan
