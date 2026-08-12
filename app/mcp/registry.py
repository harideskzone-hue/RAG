from typing import Dict, Type, Any
from pydantic import BaseModel, Field

class SearchPersonOccurrencesArgs(BaseModel):
    description: str = Field(description="Visual description of the person (e.g. 'red shirt on bike')")
    camera_id: str | None = Field(default=None, description="Optional specific camera to search")

class GetVideoClipArgs(BaseModel):
    camera_id: str = Field(description="Authorized camera ID")
    start_time: str = Field(description="ISO timestamp start")
    end_time: str = Field(description="ISO timestamp end")
    evidence_id: str = Field(description="The UUID of the person occurrence evidence this clip corresponds to")

class SearchVehicleOccurrencesArgs(BaseModel):
    description: str = Field(description="Visual description of the vehicle (e.g. 'red car', 'white van')")
    camera_id: str | None = Field(default=None, description="Optional specific camera to search")

class SearchAlertsArgs(BaseModel):
    description: str = Field(description="Description of the alert to search for (e.g. 'unauthorized access')")
    camera_id: str | None = Field(default=None, description="Optional specific camera to search")

class GetCameraMetadataArgs(BaseModel):
    camera_id: str = Field(description="Authorized camera ID to fetch metadata for")

class ToolRegistry:
    """Strict allowlist for VISTA-specific MCP tools."""
    
    _allowlist: Dict[str, Type[BaseModel]] = {
        "search_person_occurrences": SearchPersonOccurrencesArgs,
        "search_vehicle_occurrences": SearchVehicleOccurrencesArgs,
        "search_alerts": SearchAlertsArgs,
        "get_camera_metadata": GetCameraMetadataArgs,
        "get_video_clip": GetVideoClipArgs
    }
    
    @classmethod
    def validate_tool_request(cls, tool_name: str, arguments: dict[str, Any]) -> BaseModel:
        """
        Validates that a tool is on the allowlist and its arguments match the schema.
        Raises ValueError if unknown or malformed.
        """
        if tool_name not in cls._allowlist:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        
        schema_class = cls._allowlist[tool_name]
        try:
            return schema_class(**arguments)
        except Exception as e:
            raise ValueError(f"Invalid arguments for {tool_name}: {str(e)}")
