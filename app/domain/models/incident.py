from datetime import datetime
from pydantic import BaseModel, Field

class Incident(BaseModel):
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime | None = None
    related_alerts: list[str] = Field(default_factory=list)
