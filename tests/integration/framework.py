from pydantic import BaseModel
from typing import Any

# Domain model mocks representing the state transitions we want to assert
class PlannerResult(BaseModel):
    plan_id: str
    agents: list[str]
    status: str = "CREATED"

class PolicyResult(BaseModel):
    decision: str
    modified_plan: PlannerResult | None = None

class SupervisorResult(BaseModel):
    iteration_count: int
    status: str

class GraphResult(BaseModel):
    entity_count: int

class MemoryResult(BaseModel):
    episode_updated: bool

class ReasoningResult(BaseModel):
    hypothesis_generated: bool
    requires_more_info: bool = False

class EvaluationResultMock(BaseModel):
    score: float

class MockAgenticPipeline:
    """Wires up the domain abstractions to simulate an E2E run based on the IntegrationManifest."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.state_transitions = []
        
    def run(self, query: str) -> dict[str, Any]:
        results = {}
        
        # 1. Planner
        if self.manifest.planner:
            if query == "planner_fail":
                self.state_transitions.append("Planner -> Exception")
                self.state_transitions.append("Supervisor Recovery")
                return {"error": "Planner failed"}
                
            results["planner"] = PlannerResult(plan_id="p1", agents=["metadata_agent"])
            self.state_transitions.append("Execution Plan Created")
            
        # 2. Policy
        if self.manifest.policy and "planner" in results:
            if query == "policy_reject":
                results["policy"] = PolicyResult(decision="REJECT")
                self.state_transitions.append("Policy Reject")
                self.state_transitions.append("Abort")
                return results
                
            results["policy"] = PolicyResult(decision="MODIFY", modified_plan=results["planner"])
            self.state_transitions.append("Plan Modified")
            
        # 3. Supervisor
        results["supervisor"] = SupervisorResult(iteration_count=2, status="COMPLETED")
        self.state_transitions.append("Iteration Count = 2")
        
        # 4. Knowledge Graph
        if self.manifest.knowledge_graph:
            if query == "graph_empty":
                results["graph"] = GraphResult(entity_count=0)
                self.state_transitions.append("No Relationships")
                self.state_transitions.append("Gap Analyzer")
                self.state_transitions.append("Video Requested")
            else:
                results["graph"] = GraphResult(entity_count=14)
                self.state_transitions.append("Entity Count = 14")
                
        # 5. Memory
        if self.manifest.memory:
            if query == "memory_empty":
                results["memory"] = MemoryResult(episode_updated=False)
                self.state_transitions.append("No Entity")
                self.state_transitions.append("Reasoning Continues")
            else:
                results["memory"] = MemoryResult(episode_updated=True)
                self.state_transitions.append("Episode Updated")
                
        # 6. Reasoning
        if self.manifest.reasoning:
            results["reasoning"] = ReasoningResult(hypothesis_generated=True)
            self.state_transitions.append("Hypothesis Generated")
            
        # 7. Evaluation
        if self.manifest.evaluation:
            results["evaluation"] = EvaluationResultMock(score=95.0)
            self.state_transitions.append("Score Produced")
            
        return results
