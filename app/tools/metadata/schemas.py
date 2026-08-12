from typing import Any

from pydantic import BaseModel, Field


class MetadataToolResult(BaseModel):
    """
    Typed result for Metadata Tools.
    """
    success: bool = True
    rows: list[dict[str, Any]] = Field(default_factory=list)
    query_executed: str = ""
    error: str = ""
