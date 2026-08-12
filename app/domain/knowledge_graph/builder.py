from app.domain.knowledge_graph.graph import KnowledgeGraph
from app.domain.knowledge_graph.updater import GraphUpdate
from app.domain.knowledge_graph.events import GraphEventBus, EntityAdded, RelationshipCreated, EntityMerged
from app.domain.knowledge_graph.deduplication import Deduplicator
from app.domain.knowledge_graph.node import Node

class GraphBuilder:
    """Orchestrates GraphUpdate objects into the Knowledge Graph and emits GraphEvents."""
    def __init__(self, graph: KnowledgeGraph, event_bus: GraphEventBus | None = None):
        self.graph = graph
        self.event_bus = event_bus or GraphEventBus()
        self.deduplicator = Deduplicator()
        
    def apply_update(self, update: GraphUpdate):
        for node in update.new_nodes:
            self._add_or_merge_node(node)
            
        for edge in update.new_edges:
            if not self.graph.get_edge(edge.id):
                self.graph.add_edge(edge)
                self.event_bus.publish(RelationshipCreated(payload={"edge_id": str(edge.id)}))
                
        # Handle explicitly updated nodes
        for node in update.updated_nodes:
            self._add_or_merge_node(node) # Deduplicator merge handles attribute overwrites safely
            
        # Handle deletes
        for node in update.deleted_nodes:
            self.graph.remove_node(node.id)
            
        for edge in update.deleted_edges:
            self.graph.remove_edge(edge.id)
            
    def _add_or_merge_node(self, new_node: Node):
        existing_nodes = self.graph.indexes.nodes_by_type.get(new_node.type, set())
        for existing_id in existing_nodes:
            existing_node = self.graph.get_node(existing_id)
            if existing_node and self.deduplicator.should_merge(existing_node, new_node):
                merged_node = self.deduplicator.merge(existing_node, new_node)
                # Overwrite
                self.graph.add_node(merged_node)
                self.event_bus.publish(EntityMerged(payload={"source_id": str(new_node.id), "target_id": str(existing_id)}))
                return
                
        # If no merge was triggered
        self.graph.add_node(new_node)
        self.event_bus.publish(EntityAdded(payload={"node_id": str(new_node.id)}))
