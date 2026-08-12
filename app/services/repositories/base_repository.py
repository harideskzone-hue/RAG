import builtins
from abc import ABC
from typing import Any

from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool


class BaseRepository(ABC):
    """
    Base Repository pattern.
    Abstracts tool interactions and returns domain models.
    """
    def __init__(self, tool: BaseTool):
        self.tool = tool

    async def get(self, id: str, context: VistaContext) -> Any | None:
        pass

    async def list(self, context: VistaContext) -> list[Any]:
        pass

    async def search(self, query: Any, context: VistaContext) -> builtins.list[Any]:
        pass
