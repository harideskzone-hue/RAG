import uuid
from typing import Any
from app.schemas.context import ExecutionPlan, ExecutionTask

class TaskDecomposer:
    """
    Breaks down a complex query or intent into a Directed Acyclic Graph (DAG) of ExecutionTasks.
    Provides better scalability than statically assigning entire agents.
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client

    async def decompose(self, query: str, plan: ExecutionPlan) -> ExecutionPlan:
        # In MVP/Deterministic mode, we convert the static `execution_groups` or `agents` 
        # from the planner into a DAG of ExecutionTasks.
        
        tasks = []
        if not self.llm:
            from app.agents.planner.registry import AgentDependencyRegistry
            registry = AgentDependencyRegistry()
            # Deterministic fallback mapping agent groups to tasks
            task_idx = 1
            for group in plan.execution_groups:
                for agent in group:
                    # Determine dependencies based on domain registry
                    deps = registry.get_dependencies(agent)
                        
                    task_id = f"task_{task_idx}_{agent}"
                    task = ExecutionTask(
                        task_id=task_id,
                        description=f"Execute {agent}",
                        agent_type=agent,
                        dependencies=deps # Note: In a real graph, we'd map agent deps to task_ids
                    )
                    tasks.append(task)
                    task_idx += 1
            
            # Map agent dependencies to actual task dependencies
            # (Simplistic mapping for deterministic fallback)
            for t in tasks:
                mapped_deps = []
                for dep_agent in t.dependencies:
                    for earlier_t in tasks:
                        if earlier_t.agent_type == dep_agent and earlier_t.task_id != t.task_id:
                            mapped_deps.append(earlier_t.task_id)
                t.dependencies = mapped_deps
                
            plan.tasks = tasks
            return plan
        else:
            # Semantic LLM-based decomposition would go here
            # e.g., parsing query to generate independent sub-tasks
            pass
            
        return plan
