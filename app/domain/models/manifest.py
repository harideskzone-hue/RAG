from pydantic import BaseModel, Field
from app.domain.models.enums import EntityType, SchemaVersion

class AgentCapability(BaseModel):
    supported_intents: list[str] = Field(default_factory=list)
    supported_entities: list[EntityType] = Field(default_factory=list)
    supported_modalities: list[str] = Field(default_factory=list)
    supported_operations: list[str] = Field(default_factory=list)

class AgentManifest(BaseModel):
    name: str
    description: str
    capabilities: AgentCapability
    cost: str
    latency: str
    dependencies: list[str] = Field(default_factory=list)
    version: SchemaVersion = SchemaVersion.V1_0
