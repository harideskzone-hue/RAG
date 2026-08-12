from datetime import datetime
from pydantic import BaseModel

class Alert(BaseModel):
    id: str
    type: str
    camera_id: str
    timestamp: datetime
    severity: str = "low"
    resolved: bool = False
