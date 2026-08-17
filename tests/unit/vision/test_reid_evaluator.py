"""
test_reid_evaluator.py — Unit Tests for Re-ID Evaluator & Dataset Builder Guardrails
"""
import pytest
import numpy as np
from vision.re_id.reid_evaluator import compute_cmc_and_map, compute_tracklet_eval
from vision.re_id.embedding_aggregator import TrackletEmbeddingAggregator
from vision.dataset.dataset_builder import VISTADatasetBuilder


def test_compute_cmc_and_map_basic():
    # 2 Query items (Person 1, Person 2 from Camera 1)
    q_emb = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)
    q_pids = ["p1", "p2"]
    q_cids = ["cam01", "cam01"]

    # 4 Gallery items (Person 1, Person 2 from Camera 2 and Camera 1)
    g_emb = np.array([
        [0.9, 0.1, 0.0, 0.0],  # p1, cam02 -> valid match!
        [1.0, 0.0, 0.0, 0.0],  # p1, cam01 -> same camera (excluded junk!)
        [0.1, 0.9, 0.0, 0.0],  # p2, cam02 -> valid match!
        [0.0, 0.0, 1.0, 0.0],  # p3, cam02 -> different person
    ], dtype=np.float32)
    g_pids = ["p1", "p1", "p2", "p3"]
    g_cids = ["cam02", "cam01", "cam02", "cam02"]

    res = compute_cmc_and_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)

    assert res["rank1"] == 100.0
    assert res["rank5"] == 100.0
    assert res["map"] == 100.0
    assert res["num_valid_queries"] == 2


def test_compute_tracklet_eval_and_coverage():
    aggregator = TrackletEmbeddingAggregator(default_top_k=2, min_usable_threshold=0.40)

    q_tracklets = [
        {
            "person_id": "p1",
            "camera_id": "cam01",
            "embeddings": np.array([[1.0, 0.0], [0.9, 0.1]]),
            "quality_scores": [0.85, 0.90],
        },
        {
            "person_id": "p2",
            "camera_id": "cam01",
            "embeddings": np.array([[0.0, 1.0]]),
            "quality_scores": [0.10],  # Fails quality threshold!
        },
    ]

    g_tracklets = [
        {
            "person_id": "p1",
            "camera_id": "cam02",
            "embeddings": np.array([[0.95, 0.05]]),
            "quality_scores": [0.92],
        },
    ]

    res = compute_tracklet_eval(q_tracklets, g_tracklets, aggregator)

    # 1 out of 2 query tracklets was usable -> 50% query coverage, 100% gallery coverage
    # Total coverage = (1 + 1) / (2 + 1) = 66.67%
    assert 60.0 <= res["embedding_coverage_pct"] <= 70.0
    assert res["rank1"] == 100.0


def test_dataset_builder_enforces_person_id_guardrail():
    """Supervisor Guardrail 3: Dataset builder refuses to run without explicit person_id mapping."""
    with pytest.raises(ValueError, match="HARD PREREQUISITE MISSING"):
        VISTADatasetBuilder(output_dir="/tmp/test_dataset", person_id_map={})

    # Test missing track_id in mapping
    builder = VISTADatasetBuilder(
        output_dir="/tmp/test_dataset",
        person_id_map={"track_1": "person_1"}
    )
    detections = [
        {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "bbox": [10, 10, 50, 50],
            "track_id": "track_99",  # Unmapped track ID!
            "frame_id": 1,
        }
    ]
    with pytest.raises(ValueError, match="HARD PREREQUISITE VIOLATION"):
        builder.build_from_detections(detections)
