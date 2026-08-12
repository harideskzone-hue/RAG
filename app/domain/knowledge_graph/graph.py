from uuid import UUID
from typing import Any
from app.domain.knowledge_graph.node import Node
from app.domain.knowledge_graph.edge import Edge
from app.domain.knowledge_graph.indexes import GraphIndexes

class KnowledgeGraph:
    """The pure domain object for the Knowledge Graph."""
    def __init__(self):
        self.nodes: dict[UUID, Node] = {}
        self.edges: dict[UUID, Edge] = {}
        self.indexes = GraphIndexes()
        self.metadata: dict[str, Any] = {}
        
    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self.indexes.add_node(node.id, node.type)
        
    def get_node(self, node_id: UUID) -> Node | None:
        return self.nodes.get(node_id)
        
    def remove_node(self, node_id: UUID):
        node = self.nodes.get(node_id)
        if node:
            self.indexes.remove_node(node.id, node.type)
            del self.nodes[node_id]
        
    def add_edge(self, edge: Edge):
        self.edges[edge.id] = edge
        self.indexes.add_edge(edge.id, edge.type, edge.source_id, edge.target_id)
        
    def get_edge(self, edge_id: UUID) -> Edge | None:
        return self.edges.get(edge_id)
        
    def remove_edge(self, edge_id: UUID):
        edge = self.edges.get(edge_id)
        if edge:
            self.indexes.remove_edge(edge.id, edge.type, edge.source_id, edge.target_id)
            del self.edges[edge_id]
