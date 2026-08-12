from pydantic import Field

from app.domain.models import PersonMatch, VehicleMatch, AgentResult


class VectorResult(AgentResult):
    """
    Typed result specifically for the Vector Agent.
    """
    person_matches: list[PersonMatch] = Field(default_factory=list)
    vehicle_matches: list[VehicleMatch] = Field(default_factory=list)
    requires_vlm: bool = False
