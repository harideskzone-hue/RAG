import logging
from typing import Any

from app.agents.registry import agent_registry
from app.schemas.context import VistaContext

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    The Dispatcher ONLY decides how to dispatch a task to a specific agent.
    No retries, no scheduling, only dispatch.
    """

    def _resolve_agent(self, agent_name: str):
        """
        Resolve an agent by name with fallback.
        Handles the mismatch between planner short names (e.g., "metadata")
        and registered agent names (e.g., "metadata_agent").
        """
        agent = agent_registry.get_agent(agent_name)
        if agent:
            return agent

        # Fallback: try appending "_agent" suffix
        suffixed_name = f"{agent_name}_agent"
        agent = agent_registry.get_agent(suffixed_name)
        if agent:
            logger.debug(f"Resolved agent '{agent_name}' -> '{suffixed_name}'")
            return agent

        return None

    async def dispatch(self, agent_name: str, context: VistaContext) -> Any:
        agent = self._resolve_agent(agent_name)
        if not agent:
            # Check if this is an MCP tool instead of an agent
            from app.mcp.registry import ToolRegistry
            if agent_name in ToolRegistry._allowlist:
                from app.mcp.adapter import MCPToolAdapter
                adapter = MCPToolAdapter()
                # Find the tool arguments in the execution plan
                tool_req = next((t for t in context.execution_plan.tools if t.name == agent_name), None)
                arguments = tool_req.arguments if tool_req else {}
                
                # Execute the MCP tool directly
                # MCPToolAdapter handles RBAC, Schema Validation, Execution, and Normalization
                result = await adapter.execute_tool(agent_name, arguments, context)
                return result
                
            available = list(agent_registry.get_all_agents().keys())
            raise ValueError(
                f"Cannot dispatch. Agent '{agent_name}' not found in registry. "
                f"Available agents: {available}"
            )

        if not agent.validate(context):
            raise ValueError(f"Agent '{agent_name}' validation failed for this context.")

        plan = await agent.plan(context)
        result = await agent.execute(context, plan)

        if not agent.verify(result):
            raise ValueError(f"Agent '{agent_name}' result verification failed.")

        agent.finish(context, result)

        return result
