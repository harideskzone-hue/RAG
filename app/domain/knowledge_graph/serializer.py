from app.domain.knowledge_graph.graph import KnowledgeGraph
import json

class GraphSerializer:
    """Serializes the Knowledge Graph into various output formats."""
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        
    def to_json(self) -> str:
        nodes = [n.model_dump() for n in self.graph.nodes.values()]
        edges = [e.model_dump() for e in self.graph.edges.values()]
        return json.dumps({"nodes": nodes, "edges": edges}, default=str, indent=2)
        
    def to_markdown(self) -> str:
        lines = ["# Knowledge Graph\n", "## Entities"]
        for node in self.graph.nodes.values():
            lines.append(f"- **{node.type}** (ID: {node.id}): {node.attributes}")
            
        lines.append("\n## Relationships")
        for edge in self.graph.edges.values():
            source = self.graph.get_node(edge.source_id)
            target = self.graph.get_node(edge.target_id)
            s_name = f"{source.type}({source.id})" if source else str(edge.source_id)
            t_name = f"{target.type}({target.id})" if target else str(edge.target_id)
            lines.append(f"- {s_name} --[{edge.type}]--> {t_name}")
            
        return "\n".join(lines)
        
    def to_graphml(self) -> str:
        # Placeholder for future Neo4j/Gephi exports
        return "<graphml></graphml>"
