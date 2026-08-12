from abc import ABC, abstractmethod
from typing import Any

from app.schemas.context import VistaContext


class BaseTool(ABC):
    """
    Common interface for all external tools (Postgres, Milvus, Mongo, S3, Video, etc.).
    Makes tools interchangeable and testable.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the tool."""

    @abstractmethod
    async def execute(self, context: VistaContext, **kwargs) -> Any:
        """
        Execute the tool's core logic.
        """

    @abstractmethod
    def validate(self, **kwargs) -> bool:
        """
        Validate the arguments before execution.
        """

    @abstractmethod
    async def health(self) -> bool:
        """
        Check if the underlying service (e.g., PostgreSQL, Milvus) is available.
        """

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        Return metadata about the tool (e.g., supported versions, constraints).
        """
