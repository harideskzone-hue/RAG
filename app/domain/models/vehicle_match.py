from datetime import datetime
from pydantic import BaseModel

class VehicleMatch(BaseModel):
    id: str
    camera_id: str
    timestamp: datetime
    score: float
    license_plate: str | None = None
    description: str
