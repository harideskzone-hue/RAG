"""
embedding_aggregator.py — Top-K Quality-Weighted Tracklet Embedding Aggregation
==================================================================================
Performs double L2-normalized, quality-weighted aggregation across frame crops
belonging to a single object tracklet.

Mathematical Pipeline:
1. L2 normalize each frame embedding e_i -> e_hat_i
2. Filter crops where quality_score >= min_usable_threshold
3. Select Top-K highest quality crops
4. Compute quality-weighted sum: v_agg = sum(q_i * e_hat_i)
5. Final L2 normalize: e_track = v_agg / ||v_agg||_2
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class TrackletEmbeddingAggregator:
    """
    Quality-Weighted Tracklet Embedding Aggregator for Arbitrary Dimension D.
    """

    def __init__(self, default_top_k: int = 5, min_usable_threshold: float = 0.40) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be at least 1")
        self.default_top_k = default_top_k
        self.min_usable_threshold = min_usable_threshold

    def aggregate(
        self,
        embeddings: Union[List[np.ndarray], np.ndarray],
        quality_scores: Union[List[float], np.ndarray],
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Union[str, int, float, List[int]]]]:
        """
        Aggregate multi-frame crop embeddings into a single tracklet embedding.

        Args:
            embeddings: List or 2D array of frame embeddings shape (N, D)
            quality_scores: List or 1D array of quality scores shape (N,)
            top_k: Number of top-quality crops to aggregate (defaults to default_top_k)

        Returns:
            track_embedding: 1D L2-normalized numpy array shape (D,) or None if all rejected
            meta: Metadata dictionary with status, crop counts, and quality metrics
        """
        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            raise ValueError("top_k must be at least 1")

        if embeddings is None or len(embeddings) == 0:
            return None, {
                "status": "NO_USABLE_EMBEDDING",
                "reason": "Empty embeddings input",
                "total_crops": 0,
                "usable_crops": 0,
            }

        arr_emb = np.asarray(embeddings, dtype=np.float32)
        if arr_emb.ndim != 2 or arr_emb.shape[0] == 0:
            return None, {
                "status": "NO_USABLE_EMBEDDING",
                "reason": "Invalid embeddings shape, expected 2D array (N, D)",
                "total_crops": 0,
                "usable_crops": 0,
            }

        num_crops, dim = arr_emb.shape
        q_scores = np.asarray(quality_scores, dtype=np.float32).flatten()

        if q_scores.shape[0] != num_crops:
            raise ValueError(
                f"Mismatch: embeddings count ({num_crops}) != quality_scores count ({q_scores.shape[0]})"
            )

        # 1. First L2 Normalization of individual frame embeddings
        norms = np.linalg.norm(arr_emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        norm_emb = arr_emb / norms

        # 2. Filter crops by usability threshold
        usable_indices = [
            i for i, q in enumerate(q_scores)
            if q >= self.min_usable_threshold and not np.isnan(q) and not np.isinf(q)
        ]

        if not usable_indices:
            return None, {
                "status": "NO_USABLE_EMBEDDING",
                "reason": f"All {num_crops} crops failed quality threshold ({self.min_usable_threshold:.2f})",
                "total_crops": num_crops,
                "usable_crops": 0,
                "max_quality_score": round(float(np.max(q_scores)) if num_crops > 0 else 0.0, 4),
            }

        # 3. Sort usable crops by quality score descending and select Top-K
        usable_indices.sort(key=lambda idx: q_scores[idx], reverse=True)
        selected_indices = usable_indices[:k]

        selected_embeddings = norm_emb[selected_indices]  # Shape (K_sel, D)
        selected_weights = q_scores[selected_indices][:, np.newaxis]  # Shape (K_sel, 1)

        # 4. Quality-Weighted Linear Combination
        weighted_sum = np.sum(selected_embeddings * selected_weights, axis=0)  # Shape (D,)

        # 5. Final L2 Normalization of tracklet embedding
        track_norm = np.linalg.norm(weighted_sum)
        if track_norm < 1e-12:
            return None, {
                "status": "NO_USABLE_EMBEDDING",
                "reason": "Weighted sum norm is zero",
                "total_crops": num_crops,
                "usable_crops": len(usable_indices),
            }

        track_embedding = (weighted_sum / track_norm).astype(np.float32)

        meta = {
            "status": "SUCCESS",
            "dimension": dim,
            "total_crops": num_crops,
            "usable_crops": len(usable_indices),
            "aggregated_crops": len(selected_indices),
            "top_k_used": k,
            "selected_indices": [int(idx) for idx in selected_indices],
            "selected_quality_scores": [round(float(q_scores[idx]), 4) for idx in selected_indices],
            "mean_selected_quality": round(float(np.mean(q_scores[selected_indices])), 4),
        }

        return track_embedding, meta
