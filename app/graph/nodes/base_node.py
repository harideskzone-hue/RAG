from abc import ABC, abstractmethod
from typing import Any

from app.schemas.context import VistaContext


class BaseNode(ABC):
    """
    Common interface for LangGraph Nodes.
    A Node is different from an Agent; nodes encapsulate logic for the Graph (e.g., Intent Node, Evidence Node).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the node in the graph."""

    @abstractmethod
    async def __call__(self, state: VistaContext) -> dict[str, Any]:
        """
        The entrypoint for LangGraph. 
        Takes the current VistaContext and returns a dictionary with updates to the state.
        """
