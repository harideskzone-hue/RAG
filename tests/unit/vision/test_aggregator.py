"""
test_aggregator.py — Unit Tests for TrackletEmbeddingAggregator (Group D)
"""
import pytest
import numpy as np
from vision.re_id.embedding_aggregator import TrackletEmbeddingAggregator


def test_aggregator_top_k_selection_and_double_l2_norm():
    aggregator = TrackletEmbeddingAggregator(default_top_k=3, min_usable_threshold=0.40)
    dim = 512
    num_crops = 6

    # Create 6 distinct unnormalized embeddings
    rng = np.random.RandomState(42)
    raw_embeddings = rng.randn(num_crops, dim).astype(np.float32) * 5.0

    # Quality scores: 2 poor crops (< 0.40), 4 good crops
    quality_scores = [0.20, 0.95, 0.80, 0.10, 0.90, 0.60]

    track_emb, meta = aggregator.aggregate(raw_embeddings, quality_scores, top_k=3)

    assert meta["status"] == "SUCCESS"
    assert track_emb is not None
    assert track_emb.shape == (dim,)
    # Final L2 normalization guarantee
    assert abs(np.linalg.norm(track_emb) - 1.0) < 1e-5

    # Verify top-3 usable crops selected: indices 1 (0.95), 4 (0.90), 2 (0.80)
    assert meta["selected_indices"] == [1, 4, 2]
    assert meta["aggregated_crops"] == 3
    assert meta["usable_crops"] == 4


def test_aggregator_arbitrary_dimensions():
    aggregator = TrackletEmbeddingAggregator(default_top_k=2)

    for dim in [128, 512, 768, 1024]:
        rng = np.random.RandomState(dim)
        embeddings = rng.randn(4, dim).astype(np.float32)
        q_scores = [0.85, 0.75, 0.90, 0.80]

        track_emb, meta = aggregator.aggregate(embeddings, q_scores)

        assert meta["status"] == "SUCCESS"
        assert meta["dimension"] == dim
        assert track_emb.shape == (dim,)
        assert abs(np.linalg.norm(track_emb) - 1.0) < 1e-5


def test_aggregator_all_crops_rejected_returns_no_embedding():
    """Supervisor Directive: Handle all crops rejected explicitly with NO_USABLE_EMBEDDING."""
    aggregator = TrackletEmbeddingAggregator(min_usable_threshold=0.50)
    embeddings = np.random.randn(5, 512).astype(np.float32)
    # All crops fail quality threshold
    low_quality_scores = [0.10, 0.25, 0.35, 0.49, 0.05]

    track_emb, meta = aggregator.aggregate(embeddings, low_quality_scores)

    assert track_emb is None
    assert meta["status"] == "NO_USABLE_EMBEDDING"
    assert meta["usable_crops"] == 0
    assert "failed quality threshold" in meta["reason"]


def test_aggregator_empty_or_invalid_inputs():
    aggregator = TrackletEmbeddingAggregator()

    # Empty array
    track_emb, meta = aggregator.aggregate([], [])
    assert track_emb is None
    assert meta["status"] == "NO_USABLE_EMBEDDING"

    # Mismatched lengths
    embeddings = np.random.randn(3, 512).astype(np.float32)
    with pytest.raises(ValueError, match="Mismatch"):
        aggregator.aggregate(embeddings, [0.8, 0.9])
