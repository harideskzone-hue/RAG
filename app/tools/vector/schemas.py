from typing import Any

from pydantic import BaseModel, Field

from app.tools.vector.store import VectorMatch

class VectorToolResult(BaseModel):
    """
    Typed result for Vector Tools (e.g. Milvus).
    """
    success: bool = True
    matches: list[VectorMatch] = Field(default_factory=list)
    collection_searched: str = ""
    error: str = ""
