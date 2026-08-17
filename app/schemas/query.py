from pydantic import BaseModel, Field

class QueryIntent(BaseModel):
    domain: str = "investigation"
    operation: str = ""
    target_type: str = ""
    semantic_constraints: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    temporal_constraints: list[str] = Field(default_factory=list)
    spatial_constraints: list[str] = Field(default_factory=list)
    search_operations: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    conversation_reference: str | None = None
    raw_query: str = ""
