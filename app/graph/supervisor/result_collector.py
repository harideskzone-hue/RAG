import logging
import uuid
from datetime import datetime, timezone

from app.domain.evidence import EvidenceBundle, MetadataEvidence
from app.platform.tracing.context import get_conversation_id, get_execution_id
from app.schemas.context import BaseResult, VistaContext

logger = logging.getLogger(__name__)


class ResultCollector:
    """
    Collects results from individual agents and merges them into the global context.
    Handles deduplication, ordering, and provenance.
    """
    def collect(self, agent_name: str, result: BaseResult, context: VistaContext):
        context.results[agent_name] = result
        
        # EvidenceBundle is strictly for authoritative retrieval.
        # Downstream reasoning agents may emit entities/claims, but they cannot inject evidence.
        if not getattr(context, "evidence_bundle", None):
            context.evidence_bundle = EvidenceBundle()
            
        # Update context confidence_score if result carries a positive overall confidence score
        if hasattr(result, "confidence") and result.confidence:
            c = result.confidence
            conf_val = getattr(c, "overall", c) if hasattr(c, "overall") else (c if isinstance(c, (int, float)) else 0.0)
            if isinstance(conf_val, (int, float)) and conf_val > 0:
                context.confidence_score = float(conf_val)

        # Handle agents that return an EvidenceBundle
        if hasattr(result, "bundle") and getattr(result, "bundle", None):
            for ev in result.bundle.evidence:
                context.evidence_bundle.add_evidence(ev)
                
        # Handle agents that return a raw list of Evidence objects
        evidence = getattr(result, "evidence", None)
        if evidence:
            for ev in evidence:
                if hasattr(ev, "generate_hash"):
                    context.evidence_bundle.add_evidence(ev)
                else:
                    try:
                        ts = datetime.fromisoformat(ev.timestamp.replace("Z", "+00:00")) if ev.timestamp else datetime.now(timezone.utc)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to parse timestamp for evidence from {agent_name}: {e}")
                        ts = datetime.now(timezone.utc)
                        
                    adapted_ev = MetadataEvidence(
                        evidence_id=str(uuid.uuid4()),
                        source=agent_name,
                        confidence=getattr(ev, "confidence", 1.0),
                        timestamp=ts,
                        trace_id=get_execution_id() or get_conversation_id(),
                        metadata={"camera_id": getattr(ev, "camera_id", None), "description": getattr(ev, "description", "")}
                    )
                    context.evidence_bundle.add_evidence(adapted_ev)

