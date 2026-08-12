from app.domain.knowledge_graph.graph import KnowledgeGraph
from app.domain.knowledge_graph.algorithms import GraphAlgorithms

class GraphStatistics:
    """Exposes statistics for Supervisor and Reasoning logs."""
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        
    def get_stats(self) -> dict:
        components = GraphAlgorithms.connected_components(self.graph)
        return {
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "connected_components": len(components),
            "entity_types": dict((k, len(v)) for k, v in self.graph.indexes.nodes_by_type.items()),
            "relationship_types": dict((k, len(v)) for k, v in self.graph.indexes.edges_by_type.items()),
            "average_degree": (len(self.graph.edges) * 2 / len(self.graph.nodes)) if self.graph.nodes else 0
        }
