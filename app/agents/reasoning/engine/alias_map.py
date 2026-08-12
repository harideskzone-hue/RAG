import json
from uuid import UUID
from app.domain.evidence import EvidenceBundle

class EvidenceAliasMap:
    """
    Deterministically maps request-local aliases (E1, E2) to authoritative UUIDs and vice versa.
    Maintains a strict security boundary preventing LLM hallucination of evidence IDs.
    """
    def __init__(self, bundle: EvidenceBundle | None):
        self._alias_to_uuid = {}
        self._uuid_to_alias = {}
        self._evidence_details = {}
        
        if bundle and bundle.evidence:
            for i, ev in enumerate(bundle.evidence):
                alias = f"E{i+1}"
                uuid_str = str(ev.evidence_id)
                self._alias_to_uuid[alias] = uuid_str
                self._uuid_to_alias[uuid_str] = alias
                
                # Extract some human readable details for the LLM prompt
                metadata = getattr(ev, "metadata", {})
                desc = metadata.get("description", "")
                cam = metadata.get("camera_id", "Unknown Camera")
                self._evidence_details[alias] = f"[{cam}] {desc}".strip()
                
    def resolve_alias(self, alias: str) -> str:
        """
        Resolves an alias like 'E1' to its real UUID.
        Raises ValueError if the alias is unknown, preventing hallucination.
        """
        if alias not in self._alias_to_uuid:
            raise ValueError(f"Unknown evidence alias: {alias}")
        return self._alias_to_uuid[alias]
        
    def to_llm_context_string(self) -> str:
        """
        Formats the authorized evidence into a string for the LLM prompt.
        """
        if not self._alias_to_uuid:
            return "No authorized evidence available."
            
        lines = []
        for alias, details in self._evidence_details.items():
            lines.append(f"{alias} -> {details}")
        return "\n".join(lines)
