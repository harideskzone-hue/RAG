from collections import defaultdict
from uuid import UUID
from typing import Set

class GraphIndexes:
    """Maintains O(1) lookups for graph components."""
    def __init__(self):
        self.nodes_by_type: dict[str, set[UUID]] = defaultdict(set)
        self.edges_by_type: dict[str, set[UUID]] = defaultdict(set)
        self.edges_by_source: dict[UUID, set[UUID]] = defaultdict(set)
        self.edges_by_target: dict[UUID, set[UUID]] = defaultdict(set)
        # We can add Timestamp Index or Evidence Index later if needed.
        
    def add_node(self, node_id: UUID, node_type: str):
        self.nodes_by_type[node_type].add(node_id)
        
    def add_edge(self, edge_id: UUID, edge_type: str, source_id: UUID, target_id: UUID):
        self.edges_by_type[edge_type].add(edge_id)
        self.edges_by_source[source_id].add(edge_id)
        self.edges_by_target[target_id].add(edge_id)
        
    def remove_node(self, node_id: UUID, node_type: str):
        if node_id in self.nodes_by_type.get(node_type, set()):
            self.nodes_by_type[node_type].remove(node_id)
            
    def remove_edge(self, edge_id: UUID, edge_type: str, source_id: UUID, target_id: UUID):
        if edge_id in self.edges_by_type.get(edge_type, set()):
            self.edges_by_type[edge_type].remove(edge_id)
        if edge_id in self.edges_by_source.get(source_id, set()):
            self.edges_by_source[source_id].remove(edge_id)
        if edge_id in self.edges_by_target.get(target_id, set()):
            self.edges_by_target[target_id].remove(edge_id)
