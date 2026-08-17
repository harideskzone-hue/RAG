from datetime import datetime
from pydantic import BaseModel

class PersonMatch(BaseModel):
    id: str
    camera_id: str
    timestamp: datetime
    score: float
    description: str
    bbox: list[float] | None = None
    origin: dict | None = None
    attributes: dict | None = None
