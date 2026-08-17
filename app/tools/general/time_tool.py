import time
from datetime import datetime, timezone
from typing import Any

from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool


class TimeTool(BaseTool):
    """
    General System Time Tool.
    Retrieves system wall-clock date and time without triggering CCTV vector database RAG.
    """
    def __init__(self):
        self._name = "time_tool"
        self._description = "Returns current system wall-clock date, time, and timezone."

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, context: VistaContext, **kwargs) -> dict[str, Any]:
        now = datetime.now()
        now_utc = datetime.now(timezone.utc)
        
        formatted_time = now.strftime("%I:%M %p")
        formatted_date = now.strftime("%A, %B %d, %Y")
        iso_str = now_utc.isoformat()

        answer = f"The current system time is {formatted_time} ({formatted_date})."
        
        return {
            "success": True,
            "formatted_time": formatted_time,
            "formatted_date": formatted_date,
            "timestamp_iso": iso_str,
            "answer": answer
        }

    def validate(self, **kwargs) -> bool:
        return True

    async def health(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "system_clock",
            "capability": "time_query"
        }
