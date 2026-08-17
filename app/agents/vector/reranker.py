from typing import Any, TypeVar
from abc import ABC, abstractmethod
import numpy as np

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
    A basic reranker that deduplicates by candidate ID and sorts by original vector score.
    """
    async def rerank(self, original_query: str, candidates: list[T], context: VistaContext) -> list[T]:
        if not candidates:
            return []
            
        # Deduplicate candidates by unique candidate ID while preserving highest score
        unique_candidates = {}
        for cand in candidates:
            cand_id = str(getattr(cand, 'id', getattr(cand, 'detection_id', str(cand))))
            existing = unique_candidates.get(cand_id)
            if not existing or getattr(cand, 'score', 0.0) > getattr(existing, 'score', 0.0):
                unique_candidates[cand_id] = cand

        deduped = list(unique_candidates.values())
        return sorted(deduped, key=lambda x: getattr(x, 'score', 0.0), reverse=True)


class SemanticReranker(BaseReranker):
    """
    Semantic reranker that rescores candidate descriptions using cosine similarity against query embedding.
    """
    def __init__(self, encoder=None):
        self.encoder = encoder

    async def rerank(self, original_query: str, candidates: list[T], context: VistaContext) -> list[T]:
        if not candidates:
            return []

        # Deduplicate by unique candidate ID first
        unique_candidates = {}
        for cand in candidates:
            cand_id = str(getattr(cand, 'id', getattr(cand, 'detection_id', str(cand))))
            existing = unique_candidates.get(cand_id)
            if not existing or getattr(cand, 'score', 0.0) > getattr(existing, 'score', 0.0):
                unique_candidates[cand_id] = cand

        deduped = list(unique_candidates.values())

        if self.encoder is None:
            from app.tools.vector.encoder import get_vector_encoder
            self.encoder = get_vector_encoder()

        try:
            query_emb = np.array(self.encoder.encode(original_query))
            q_norm = np.linalg.norm(query_emb)
            if q_norm == 0:
                q_norm = 1e-10

            for cand in deduped:
                desc = getattr(cand, 'description', '')
                if desc:
                    cand_emb = np.array(self.encoder.encode(desc))
                    c_norm = np.linalg.norm(cand_emb)
                    if c_norm == 0:
                        c_norm = 1e-10
                    sim = float(np.dot(query_emb, cand_emb) / (q_norm * c_norm))
                    # Combine original score and semantic rescore
                    cand.score = max(0.0, float(0.5 * getattr(cand, 'score', 0.0) + 0.5 * sim))
        except Exception as e:
            import logging
            logging.warning(f"Error during reranking semantic rescore: {e}")

        return sorted(deduped, key=lambda x: getattr(x, 'score', 0.0), reverse=True)
