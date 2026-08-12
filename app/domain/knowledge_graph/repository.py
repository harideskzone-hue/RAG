from typing import Protocol
from app.domain.knowledge_graph.graph import KnowledgeGraph

class GraphRepository(Protocol):
    """Persistence abstraction for the Knowledge Graph."""
    
    def load(self, graph_id: str) -> KnowledgeGraph:
        ...
        
    def save(self, graph_id: str, graph: KnowledgeGraph) -> None:
        ...

class InMemoryRepository(GraphRepository):
    """In-Memory implementation of the GraphRepository."""
    _storage: dict[str, KnowledgeGraph] = {}
    
    def __init__(self):
        pass
        
    def load(self, graph_id: str) -> KnowledgeGraph:
        if graph_id not in self._storage:
            self._storage[graph_id] = KnowledgeGraph()
        return self._storage[graph_id]
        
    def save(self, graph_id: str, graph: KnowledgeGraph) -> None:
        self._storage[graph_id] = graph
