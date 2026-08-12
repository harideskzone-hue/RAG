from pydantic import BaseModel, Field

class Camera(BaseModel):
    id: str
    location: str
    status: str
    firmware_version: str | None = None
    capabilities: list[str] = Field(default_factory=list)
