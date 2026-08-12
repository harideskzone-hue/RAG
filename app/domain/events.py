from app.graph.supervisor.telemetry import AgentEvent


class EvidenceEvent(AgentEvent):
    """Base class for domain events related to Evidence."""

class EvidenceCreated(EvidenceEvent):
    def __init__(self, **kwargs):
        kwargs['event_type'] = "EVIDENCE_CREATED"
        super().__init__(**kwargs)

class EvidenceMerged(EvidenceEvent):
    def __init__(self, **kwargs):
        kwargs['event_type'] = "EVIDENCE_MERGED"
        super().__init__(**kwargs)

class EvidenceVerified(EvidenceEvent):
    def __init__(self, **kwargs):
        kwargs['event_type'] = "EVIDENCE_VERIFIED"
        super().__init__(**kwargs)

class EvidenceRejected(EvidenceEvent):
    def __init__(self, **kwargs):
        kwargs['event_type'] = "EVIDENCE_REJECTED"
        super().__init__(**kwargs)

class EvidenceRemoved(EvidenceEvent):
    def __init__(self, **kwargs):
        kwargs['event_type'] = "EVIDENCE_REMOVED"
        super().__init__(**kwargs)
