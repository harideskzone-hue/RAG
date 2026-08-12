from typing import Any, TypeVar
from abc import ABC, abstractmethod

from app.schemas.context import VistaContext

T = TypeVar('T')

class BaseReranker(ABC):
    """
    Interface for reranking candidate pools retrieved from vector search.
    """
    
    @abstractmethod
    async def rerank(self, original_query: str, candidates: list[T], context: VistaContext) -> list[T]:
        """
        Rerank a pool of candidates based on the original query.
        """
        pass

class PassThroughReranker(BaseReranker):
    """
    A basic reranker that simply returns the candidates sorted by their original vector score.
    Used for benchmarking 'Expansion + vector' vs 'Expansion + vector + reranking'.
    """
    async def rerank(self, original_query: str, candidates: list[T], context: VistaContext) -> list[T]:
        # Sort candidates strictly by original score descending
        return sorted(candidates, key=lambda x: getattr(x, 'score', 0.0), reverse=True)
