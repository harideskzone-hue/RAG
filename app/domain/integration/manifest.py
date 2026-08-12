from pydantic import BaseModel

class IntegrationManifest(BaseModel):
    """Declarative manifest to toggle subsystems during integration runs."""
    planner: bool = True
    policy: bool = True
    knowledge_graph: bool = True
    memory: bool = True
    reasoning: bool = True
    evaluation: bool = True
    guardrails: bool = False
