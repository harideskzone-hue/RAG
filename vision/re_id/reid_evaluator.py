"""
reid_evaluator.py — Standard Person Re-ID Evaluator (Rank-1, Rank-5, mAP)
==========================================================================
Computes standard Person Re-ID metrics (Rank-1, Rank-5, Rank-10, mAP) across
Cross-Camera Query and Gallery sets with strict same-camera exclusion.
Also computes tracklet-level retrieval performance and embedding coverage %.
"""
from typing import Dict, List, Tuple, Union
import numpy as np


def compute_cmc_and_map(
    query_embeddings: np.ndarray,
    query_person_ids: Union[List[str], np.ndarray],
    query_camera_ids: Union[List[str], np.ndarray],
    gallery_embeddings: np.ndarray,
    gallery_person_ids: Union[List[str], np.ndarray],
    gallery_camera_ids: Union[List[str], np.ndarray],
    max_rank: int = 10,
) -> Dict[str, float]:
    """
    Computes Cumulative Matching Characteristics (CMC) and Mean Average Precision (mAP).

    Excludes gallery images with the SAME person_id AND SAME camera_id as query (standard Re-ID evaluation protocol).

    Args:
        query_embeddings: 2D numpy array shape (N_q, D), L2-normalized
        query_person_ids: List/array of person IDs for query items
        query_camera_ids: List/array of camera IDs for query items
        gallery_embeddings: 2D numpy array shape (N_g, D), L2-normalized
        gallery_person_ids: List/array of person IDs for gallery items
        gallery_camera_ids: List/array of camera IDs for gallery items
        max_rank: Maximum rank position for CMC curve

    Returns:
        dict containing 'rank1', 'rank5', 'rank10', 'map', and 'num_valid_queries'
    """
    q_emb = np.asarray(query_embeddings, dtype=np.float32)
    g_emb = np.asarray(gallery_embeddings, dtype=np.float32)

    q_pids = np.asarray(query_person_ids)
    q_cids = np.asarray(query_camera_ids)
    g_pids = np.asarray(gallery_person_ids)
    g_cids = np.asarray(gallery_camera_ids)

    num_q = q_emb.shape[0]
    num_g = g_emb.shape[0]

    if num_q == 0 or num_g == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "map": 0.0, "num_valid_queries": 0}

    # Cosine Similarity Matrix (N_q, N_g)
    sim_matrix = np.dot(q_emb, g_emb.T)

    cmc = np.zeros(max_rank, dtype=np.float32)
    ap_list = []
    valid_queries = 0

    for i in range(num_q):
        q_pid = q_pids[i]
        q_cid = q_cids[i]

        # Valid gallery mask: exclude same person AND same camera
        valid_gallery_mask = ~((g_pids == q_pid) & (g_cids == q_cid))
        good_mask = (g_pids == q_pid) & valid_gallery_mask

        if not np.any(good_mask):
            continue  # Query has no valid cross-camera gallery target

        valid_queries += 1

        # Slice gallery items for valid evaluation
        sub_g_pids = g_pids[valid_gallery_mask]
        sub_scores = sim_matrix[i][valid_gallery_mask]

        # Sort gallery items in descending order of similarity
        order = np.argsort(-sub_scores)
        matches = (sub_g_pids[order] == q_pid)

        # 1. CMC Calculation
        first_match_idx = np.where(matches)[0][0]
        if first_match_idx < max_rank:
            cmc[first_match_idx:] += 1.0

        # 2. Average Precision (mAP) Calculation
        num_pos = np.sum(good_mask)
        raw_cmc = matches.astype(np.float32)
        cumulative_pos = np.cumsum(raw_cmc)
        ranks = np.arange(1, len(matches) + 1)
        precision = cumulative_pos / ranks
        ap = np.sum(precision * raw_cmc) / num_pos
        ap_list.append(ap)

    if valid_queries == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "map": 0.0, "num_valid_queries": 0}

    cmc = cmc / valid_queries
    mean_ap = float(np.mean(ap_list)) if ap_list else 0.0

    return {
        "rank1": round(float(cmc[0] * 100.0), 2),
        "rank5": round(float(cmc[min(4, max_rank - 1)] * 100.0), 2),
        "rank10": round(float(cmc[min(9, max_rank - 1)] * 100.0), 2),
        "map": round(float(mean_ap * 100.0), 2),
        "num_valid_queries": valid_queries,
    }


def compute_tracklet_eval(
    query_tracklets: List[Dict],
    gallery_tracklets: List[Dict],
    aggregator,
    top_k: int = 5,
) -> Dict[str, Union[float, int]]:
    """
    Computes Tracklet-Level Rank-1, Rank-5, mAP, and Tracklet Embedding Coverage %.

    Args:
        query_tracklets: List of dicts with 'person_id', 'camera_id', 'embeddings', 'quality_scores'
        gallery_tracklets: List of dicts with 'person_id', 'camera_id', 'embeddings', 'quality_scores'
        aggregator: TrackletEmbeddingAggregator instance
        top_k: Top-K crops for quality-weighted aggregation

    Returns:
        dict with rank1, rank5, map, embedding_coverage_pct
    """
    q_vecs, q_pids, q_cids = [], [], []
    valid_q_tracklets = 0
    total_q_tracklets = len(query_tracklets)

    for item in query_tracklets:
        vec, meta = aggregator.aggregate(item["embeddings"], item["quality_scores"], top_k=top_k)
        if vec is not None:
            q_vecs.append(vec)
            q_pids.append(item["person_id"])
            q_cids.append(item["camera_id"])
            valid_q_tracklets += 1

    g_vecs, g_pids, g_cids = [], [], []
    valid_g_tracklets = 0
    total_g_tracklets = len(gallery_tracklets)

    for item in gallery_tracklets:
        vec, meta = aggregator.aggregate(item["embeddings"], item["quality_scores"], top_k=top_k)
        if vec is not None:
            g_vecs.append(vec)
            g_pids.append(item["person_id"])
            g_cids.append(item["camera_id"])
            valid_g_tracklets += 1

    coverage_q = (valid_q_tracklets / max(1, total_q_tracklets)) * 100.0
    coverage_g = (valid_g_tracklets / max(1, total_g_tracklets)) * 100.0
    total_coverage = ((valid_q_tracklets + valid_g_tracklets) / max(1, total_q_tracklets + total_g_tracklets)) * 100.0

    if not q_vecs or not g_vecs:
        return {
            "rank1": 0.0,
            "rank5": 0.0,
            "rank10": 0.0,
            "map": 0.0,
            "embedding_coverage_pct": round(total_coverage, 2),
            "valid_query_tracklets": valid_q_tracklets,
            "valid_gallery_tracklets": valid_g_tracklets,
        }

    res = compute_cmc_and_map(
        np.array(q_vecs), q_pids, q_cids,
        np.array(g_vecs), g_pids, g_cids,
    )

    res["embedding_coverage_pct"] = round(total_coverage, 2)
    res["valid_query_tracklets"] = valid_q_tracklets
    res["valid_gallery_tracklets"] = valid_g_tracklets
    return res
