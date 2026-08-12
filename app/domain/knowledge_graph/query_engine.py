from uuid import UUID
from app.domain.knowledge_graph.graph import KnowledgeGraph
from app.domain.knowledge_graph.algorithms import GraphAlgorithms
from app.domain.knowledge_graph.node import Node

class GraphQueryEngine:
    """
    Domain-specific query engine that acts as the interface for the Reasoning Engine.
    Abstracts away raw graph traversal by relying on `algorithms.py`.
    """
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        
    def find_person_path(self, person_node_id: UUID, target_location_id: UUID) -> list:
        """Finds the path a person took to reach a location."""
        return GraphAlgorithms.shortest_path(self.graph, person_node_id, target_location_id)
        
    def find_temporal_conflicts(self, entity_id: UUID) -> list[tuple[Node, Node]]:
        """
        Detects if a single entity appears in two locations at the exact same time 
        or traveling faster than physically possible (simplified for MVP).
        """
        # Get all Event nodes related to this entity
        events = []
        for edge_id in self.graph.indexes.edges_by_source.get(entity_id, set()):
            edge = self.graph.get_edge(edge_id)
            if edge and edge.type == "participates_in":
                target_node = self.graph.get_node(edge.target_id)
                if target_node and target_node.type == "Event":
                    events.append(target_node)
                    
        # Sort by timestamp
        events_with_time = [e for e in events if "timestamp" in e.attributes]
        events_with_time.sort(key=lambda n: n.attributes["timestamp"])
        
        conflicts = []
        # Find overlapping or identical timestamps for different events
        for i in range(len(events_with_time) - 1):
            e1 = events_with_time[i]
            e2 = events_with_time[i+1]
            if e1.attributes["timestamp"] == e2.attributes["timestamp"] and e1.id != e2.id:
                conflicts.append((e1, e2))
                
        return conflicts
        
    def find_missing_exits(self) -> list[Node]:
        """Finds entities that entered a location but have no exit event."""
        # Domain logic: An entity has an 'enters' relationship but no corresponding 'exits'
        # For MVP, stub this as querying the 'Event' nodes.
        pass
