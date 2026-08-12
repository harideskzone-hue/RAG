from typing import Any

from pydantic import Field

from app.agents.intent.enums import Intent
from app.schemas.context import BaseResult


class IntentResult(BaseResult):
    intent: Intent = Intent.UNKNOWN
    entities: dict[str, Any] = Field(default_factory=dict)
    query_intent: Any | None = None
    requires_clarification: bool = False
    missing_entities: list[str] = Field(default_factory=list)
    requires_vlm: bool = False
