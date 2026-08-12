
from langgraph.graph import StateGraph

from app.memory.manager import MemoryManager
from app.schemas.context import VistaContext


class GraphBuilder:
    """
    Builder for the main VISTA AI LangGraph workflow.
    
    ARCHITECTURE NOTE (Phase 3 Validation):
    ----------------------------------------
    Currently, the system routes directly from the chat API route to the Supervisor,
    bypassing LangGraph orchestration entirely. This is intentional during the
    stabilization phase.
    
    The Supervisor internally handles: Intent → Planner → Validator → Agent Dispatch
    
    When LangGraph orchestration is needed (e.g., for complex multi-step workflows
    with conditional routing, human-in-the-loop, or checkpointing), this builder
    should be wired into the application via a dependency that replaces the direct
    Supervisor call in the chat route.
    
    To activate LangGraph:
    1. Implement node wrappers in app/graph/nodes/ (IntentNode, PlannerNode, etc.)
    2. Implement routing logic in app/graph/edges/
    3. Call build_core_workflow() to wire the graph
    4. Replace get_supervisor DI with a compiled graph runner
    """
    def __init__(self, memory_manager: MemoryManager | None = None, checkpointer=None):
        self.workflow = StateGraph(VistaContext)
        self.memory_manager = memory_manager
        self.checkpointer = checkpointer

    def add_node(self, name: str, action):
        """Add a node to the workflow."""
        self.workflow.add_node(name, action)
        return self

    def add_edge(self, start: str, end: str):
        """Add an edge between nodes."""
        self.workflow.add_edge(start, end)
        return self

    def add_conditional_edges(self, start: str, router, paths: dict):
        """Add conditional edges for dynamic routing."""
        self.workflow.add_conditional_edges(start, router, paths)
        return self

    def compile(self):
        """Compile the graph into a runnable application."""
        return self.workflow.compile(checkpointer=self.checkpointer)

    def build_core_workflow(self):
        """
        Builds the foundational workflow:
        User → Intent → Planner → Workflow Validator → Supervisor → Memory → END
        
        NOT YET ACTIVE — requires node implementations in app/graph/nodes/.
        See class docstring for activation steps.
        """
        # Placeholder — activate when node implementations are ready.
        # Required nodes: IntentNode, PlannerNode, ValidatorNode, SupervisorNode
        # Required edges: intent→planner, planner→validator, validator→supervisor
        # Conditional edges: validator can route to clarification or rejection
        return self

