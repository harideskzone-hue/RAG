from datetime import datetime, timedelta, timezone

from app.memory.metrics import MemoryMetrics
from app.memory.policy import MemoryPolicy
from app.schemas.context import VistaContext


class EvictionPolicy:
    """
    Handles eviction of evidence and other structured state to manage memory size.
    Enforces Priority and TTL rules.
    """
    def __init__(self, policy: MemoryPolicy, metrics: MemoryMetrics):
        self.policy = policy
        self.metrics = metrics

    def execute(self, context: VistaContext):
        """
        Applies eviction rules to the context in place.
        """
        if not self.policy.enable_eviction:
            return

        now = datetime.now(timezone.utc)
        
        # 1. Evict old evidence from EvidenceBundle
        if context.evidence_bundle and context.evidence_bundle.evidence:
            retained_evidence = []
            evicted_count = 0
            
            for ev in context.evidence_bundle.evidence:
                # Keep high-priority evidence regardless of TTL
                priority = ev.metadata.get("priority", "normal").lower()
                if priority in ["critical", "high"]:
                    retained_evidence.append(ev)
                    continue
                    
                # Apply TTL for routine evidence
                if self.policy.evidence_ttl_hours:
                    age = now - ev.timestamp
                    if age > timedelta(hours=self.policy.evidence_ttl_hours):
                        evicted_count += 1
                        continue # Evict
                
                retained_evidence.append(ev)
                
            context.evidence_bundle.evidence = retained_evidence
            self.metrics.increment_eviction(evicted_count)

        # 2. Result Memory (e.g. Reports) - Permanent by default, or TTL if specified
        if self.policy.report_ttl_hours and "report_agent" in context.results:
            # If we had a mechanism to age out reports from the state, we would do it here.
            pass
