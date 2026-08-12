from uuid import UUID
from app.domain.knowledge_graph.graph import KnowledgeGraph
from app.domain.knowledge_graph.node import Node
from app.domain.knowledge_graph.edge import Edge
from collections import deque

class GraphAlgorithms:
    """Generic mathematical graph operations devoid of domain logic."""
    
    @staticmethod
    def shortest_path(graph: KnowledgeGraph, source_id: UUID, target_id: UUID) -> list[Node | Edge]:
        """Finds the shortest unweighted path using BFS."""
        if source_id not in graph.nodes or target_id not in graph.nodes:
            return []
            
        queue = deque([(source_id, [])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            if current_id == target_id:
                return path
                
            # Iterate neighbors
            for edge_id in graph.indexes.edges_by_source.get(current_id, set()):
                edge = graph.get_edge(edge_id)
                if not edge: continue
                next_id = edge.target_id
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [edge, graph.get_node(next_id)]))
                    
            for edge_id in graph.indexes.edges_by_target.get(current_id, set()):
                edge = graph.get_edge(edge_id)
                if not edge: continue
                next_id = edge.source_id
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [edge, graph.get_node(next_id)]))
                    
        return []

    @staticmethod
    def temporal_traversal(graph: KnowledgeGraph, start_time: float, end_time: float) -> list[Node]:
        """Returns nodes that fall within the given temporal window based on timestamp attribute."""
        results = []
        for node in graph.nodes.values():
            ts = node.attributes.get("timestamp")
            if ts is not None and isinstance(ts, (int, float)):
                if start_time <= ts <= end_time:
                    results.append(node)
        # Sort chronologically
        results.sort(key=lambda n: n.attributes.get("timestamp", 0))
        return results
        
    @staticmethod
    def connected_components(graph: KnowledgeGraph) -> list[set[UUID]]:
        """Returns a list of connected components (sets of Node UUIDs)."""
        visited = set()
        components = []
        
        for node_id in graph.nodes:
            if node_id not in visited:
                component = set()
                queue = deque([node_id])
                while queue:
                    current_id = queue.popleft()
                    if current_id not in component:
                        component.add(current_id)
                        visited.add(current_id)
                        # Neighbors
                        for edge_id in graph.indexes.edges_by_source.get(current_id, set()):
                            edge = graph.get_edge(edge_id)
                            if edge: queue.append(edge.target_id)
                        for edge_id in graph.indexes.edges_by_target.get(current_id, set()):
                            edge = graph.get_edge(edge_id)
                            if edge: queue.append(edge.source_id)
                components.append(component)
        return components
