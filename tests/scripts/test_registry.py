from app.agents.registry import agent_registry
print("Registry in test_registry:", id(agent_registry), list(agent_registry.get_all_agents().keys()))

from app.graph.supervisor.dispatcher import Dispatcher
from app.agents.registry import agent_registry as disp_reg
print("Registry in dispatcher:", id(disp_reg), list(disp_reg.get_all_agents().keys()))
