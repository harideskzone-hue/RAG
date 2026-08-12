from app.domain.knowledge_graph.node import Node

class DeduplicationProfile:
    """Configurable scoring weights for entity resolution."""
    def __init__(self, weights: dict[str, float], threshold: float = 0.8):
        self.weights = weights
        self.threshold = threshold

class Deduplicator:
    """
    Deterministic entity resolution using entity-specific scoring profiles
    instead of relying on exact attribute equality.
    """
    def __init__(self):
        self.profiles = {
            "Person": DeduplicationProfile({"face_id": 1.0, "name": 0.4, "camera": 0.2, "timestamp": 0.1}, threshold=0.8),
            "Vehicle": DeduplicationProfile({"plate": 1.0, "vehicle_id": 1.0, "color": 0.3, "model": 0.4, "timestamp": 0.1}, threshold=0.8),
            "Camera": DeduplicationProfile({"camera_id": 1.0, "location": 0.8}, threshold=1.0),
            "Event": DeduplicationProfile({"event_id": 1.0, "timestamp": 0.4, "camera": 0.3}, threshold=1.0)
        }
        
    def score_match(self, node1: Node, node2: Node) -> float:
        if node1.type != node2.type:
            return 0.0
            
        profile = self.profiles.get(node1.type)
        if not profile:
            # Fallback to strict exact match if no profile
            return 1.0 if node1.id == node2.id else 0.0
            
        score = 0.0
        for attr, weight in profile.weights.items():
            if attr in node1.attributes and attr in node2.attributes:
                if node1.attributes[attr] == node2.attributes[attr]:
                    score += weight
                    
        return score
        
    def should_merge(self, node1: Node, node2: Node) -> bool:
        if node1.id == node2.id:
            return True
        if node1.type != node2.type:
            return False
            
        profile = self.profiles.get(node1.type)
        if not profile:
            return node1.id == node2.id
            
        return self.score_match(node1, node2) >= profile.threshold
        
    def merge(self, node1: Node, node2: Node) -> Node:
        # Merge attributes, preferring node1's existing attributes on collision, then node2's
        merged_attrs = {**node2.attributes, **node1.attributes}
        return Node(
            id=node1.id, # Keep existing ID
            type=node1.type,
            attributes=merged_attrs,
            confidence=max(node1.confidence, node2.confidence),
            created_at=min(node1.created_at, node2.created_at) if node1.created_at and node2.created_at else (node1.created_at or node2.created_at)
        )
