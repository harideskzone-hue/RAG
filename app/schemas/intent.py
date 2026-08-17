from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class IntentType(Enum):
    SEARCH = "SEARCH"
    COUNT = "COUNT"
    TIMELINE = "TIMELINE"
    RELATIONSHIP = "RELATIONSHIP"
    UNKNOWN = "UNKNOWN"

class EntityType(Enum):
    PERSON = "PERSON"
    VEHICLE = "VEHICLE"
    OBJECT = "OBJECT"
    UNKNOWN = "UNKNOWN"

class TemporalConstraints(BaseModel):
    start_time_str: Optional[str] = Field(None, description="Start time in ISO format or descriptive (e.g. '8 PM')")
    end_time_str: Optional[str] = Field(None, description="End time in ISO format or descriptive")
    is_relative: bool = Field(False, description="Whether the time is relative (e.g. 'last hour')")
    
class SpatialConstraints(BaseModel):
    camera_ids: List[str] = Field(default_factory=list, description="Explicit camera IDs if mentioned")
    locations: List[str] = Field(default_factory=list, description="Descriptive locations (e.g. 'entrance')")
    
class QueryIntent(BaseModel):
    """
    The structured semantic representation of a user's natural language query.
    This MUST be generated purely by the LLM without any hardcoded keyword regexes.
    """
    intent_type: IntentType = Field(description="The primary action of the query")
    entity_type: EntityType = Field(description="The primary entity being queried")
    
    identity_target: Optional[str] = Field(None, description="Specific ID if requested (e.g. P000123)")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Attributes like color, clothing")
    
    temporal_constraints: TemporalConstraints = Field(default_factory=TemporalConstraints)
    spatial_constraints: SpatialConstraints = Field(default_factory=SpatialConstraints)
    
    relationships: List[str] = Field(default_factory=list, description="Relationships like 'subsequent movement'")
    
    aggregation: Optional[str] = Field(None, description="e.g. 'sum', 'average' for counting")
    ordering: Optional[str] = Field(None, description="e.g. 'chronological', 'descending'")
    
    requested_evidence: List[str] = Field(default_factory=list, description="What the user wants to see (e.g. 'identity', 'timeline')")
    answer_strategy: str = Field("direct_answer", description="How the LLM should format the response")
    
    confidence: float = Field(0.0, description="LLM's self-assessed confidence in parsing this intent (0.0 to 1.0)")
    
    is_valid: bool = Field(True, description="False if the query is nonsensical or completely unparseable")
    clarification_needed: Optional[str] = Field(None, description="Message to the user if query is ambiguous")
