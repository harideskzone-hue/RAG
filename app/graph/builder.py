
from langgraph.graph import StateGraph, END

from app.memory.manager import MemoryManager
from app.schemas.context import VistaContext


class GraphBuilder:
    """
    Builder for the main VISTA AI LangGraph workflow.
    
    The production graph implements:
        Intent → Planner → Retrieval → Verification → Response → Grounding → END
    
    With ABSTAIN routing at:
        - Intent failure → ABSTAIN
        - Verification failure (no evidence) → ABSTAIN
        - Grounding failure (hallucination) → REJECT/ABSTAIN
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

    def build_core_workflow(self, intent_node, planner_node, retrieval_node, 
                            verification_node, response_node, grounding_node):
        """
        Builds the production VISTA Agentic RAG workflow:
        
            Intent → Planner → Retrieval → Verification → Response → Grounding → END
        
        ABSTAIN paths:
            Intent (low confidence / invalid) → Response (abstain) → END
            Verification (no evidence) → Response (abstain) → END
            Grounding (hallucination) → END (with rejection message)
        """
        # Register nodes
        self.workflow.add_node("intent", intent_node.execute)
        self.workflow.add_node("planner", planner_node.execute)
        self.workflow.add_node("retrieval", retrieval_node.execute)
        self.workflow.add_node("verification", verification_node.execute)
        self.workflow.add_node("response", response_node.execute)
        self.workflow.add_node("grounding", grounding_node.execute)

        # Set entry point
        self.workflow.set_entry_point("intent")

        # Intent → route based on whether intent was parsed successfully
        def route_after_intent(state):
            if state.get("abstain_reason") or state.get("query_intent") is None:
                return "response"  # Skip to response for abstain message
            return "planner"

        self.workflow.add_conditional_edges(
            "intent",
            route_after_intent,
            {"planner": "planner", "response": "response"}
        )

        # Planner → Retrieval (always, planner handles its own abstain logic)
        self.workflow.add_edge("planner", "retrieval")

        # Retrieval → Verification
        self.workflow.add_edge("retrieval", "verification")

        # Verification → route based on evidence sufficiency
        def route_after_verification(state):
            if state.get("abstain_reason") or state.get("verified_contract") is None:
                return "response"  # No evidence → abstain response
            return "response"  # Evidence found → generate response

        self.workflow.add_conditional_edges(
            "verification",
            route_after_verification,
            {"response": "response"}
        )

        # Response → Grounding (only if we have a verified contract, not abstaining)
        def route_after_response(state):
            if state.get("abstain_reason") and state.get("verified_contract") is None:
                return END  # Pure abstain, no need to ground
            return "grounding"

        self.workflow.add_conditional_edges(
            "response",
            route_after_response,
            {"grounding": "grounding", END: END}
        )

        # Grounding → END
        self.workflow.add_edge("grounding", END)

        return self
