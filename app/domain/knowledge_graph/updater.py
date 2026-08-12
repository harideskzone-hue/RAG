from pydantic import BaseModel, Field
from app.domain.knowledge_graph.node import Node
from app.domain.knowledge_graph.edge import Edge

class GraphUpdate(BaseModel):
    """
    Schema for mutations to the Knowledge Graph.
    Ensures the GraphBuilder isn't tightly coupled to AgentResult.
    """
    new_nodes: list[Node] = Field(default_factory=list)
    new_edges: list[Edge] = Field(default_factory=list)
    updated_nodes: list[Node] = Field(default_factory=list)
    deleted_edges: list[Edge] = Field(default_factory=list)
    deleted_nodes: list[Node] = Field(default_factory=list)
