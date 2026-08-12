from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class InvestigationBlackboard(BaseModel):
    """
    Shared memory space accessible by all agents and the supervisor.
    Stores the accumulating context of the current investigation to avoid duplicate work.
    """
    id: UUID = Field(default_factory=uuid4)
    known_entities: list[UUID] = Field(default_factory=list)
    known_relationships: list[UUID] = Field(default_factory=list)
    hypotheses: list[Any] = Field(default_factory=list)
    rejected_hypotheses: list[Any] = Field(default_factory=list)
    requested_agents: list[str] = Field(default_factory=list)
    completed_actions: list[str] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(default_factory=list)
    
    def add_requested_agent(self, agent_name: str) -> None:
        if agent_name not in self.requested_agents:
            self.requested_agents.append(agent_name)
            
    def add_completed_action(self, action: str) -> None:
        if action not in self.completed_actions:
            self.completed_actions.append(action)
