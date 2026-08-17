#!/usr/bin/env python3
"""
VISTA AI — Re-ID Benchmark Diagnostic & Pair Inspection Tool

1. Independently calculates:
     - Vector dimension
     - L2 Norm A, L2 Norm B
     - Dot product
     - Cosine similarity
2. Inspects 20-50 explicit SAME and DIFFERENT pairs.
3. Diagnoses why different-person pairs were scoring closely.
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.cv.reid.osnet import OSNetExtractor


def debug_reid_pairs():
    print("\n" + "═" * 80)
    print("      VISTA AI — RE-ID PAIRWISE DIAGNOSTIC & MATHEMATICAL AUDIT")
    print("═" * 80)

    extractor = OSNetExtractor()
    reid_root = Path(PROJECT_ROOT) / "dataset" / "reid"

    # Collect sample crops per labeled identity
    identities = {}
    for split in ["train", "val", "test"]:
        sdir = reid_root / split
        if not sdir.exists():
            continue
        for pdir in sorted(sdir.iterdir()):
            if pdir.is_dir():
                crops = sorted(list(pdir.glob("*.jpg")))
                if crops:
                    identities[pdir.name] = crops

    print(f"Loaded {len(identities)} labeled ground-truth identities: {list(identities.keys())}\n")

    # Extract embeddings for first 2 crops of each identity
    emb_data = {}
    for pid, crop_list in identities.items():
        emb_data[pid] = []
        for cf in crop_list[:3]:
            img = cv2.imread(str(cf))
            if img is not None:
                # Raw feature from extractor
                raw_emb = np.array(extractor.extract(img), dtype=np.float64)
                norm = np.linalg.norm(raw_emb)
                emb_data[pid].append({
                    "path": cf,
                    "crop_name": cf.name,
                    "raw_emb": raw_emb,
                    "norm": norm,
                    "unit_emb": raw_emb / norm if norm > 0 else raw_emb
                })

    print("─" * 80)
    print("1. INDIVIDUAL EMBEDDING INTEGRITY CHECK")
    print("─" * 80)
    for pid, items in emb_data.items():
        sample = items[0]
        print(f"  Identity: {pid:<6} | Dim: {len(sample['raw_emb'])} | L2 Norm: {sample['norm']:.4f} | Vector Stats: min={np.min(sample['unit_emb']):.4f}, max={np.max(sample['unit_emb']):.4f}, mean={np.mean(sample['unit_emb']):.4f}")

    print("\n" + "─" * 80)
    print("2. SAME-PERSON PAIRWISE VERIFICATION (Expected: HIGH similarity > 0.85)")
    print("─" * 80)
    print(f" {'Pair #':<8} | {'Identity A':<10} | {'Identity B':<10} | {'Norm A':<8} | {'Norm B':<8} | {'Dot Product':<12} | {'Cosine Sim':<11} | {'Status'}")
    print("─" * 80)

    pair_idx = 1
    same_scores = []
    for pid, items in emb_data.items():
        if len(items) >= 2:
            item_a = items[0]
            item_b = items[1]
            dot = float(np.dot(item_a['unit_emb'], item_b['unit_emb']))
            cos_sim = dot / (np.linalg.norm(item_a['unit_emb']) * np.linalg.norm(item_b['unit_emb']))
            same_scores.append(cos_sim)
            status = "MATCH (CORRECT)" if cos_sim >= 0.72 else "LOW (FALSE SPLIT)"
            print(f" {pair_idx:<8d} | {pid:<10} | {pid:<10} | {item_a['norm']:<8.2f} | {item_b['norm']:<8.2f} | {dot:<12.4f} | {cos_sim:<11.4f} | {status}")
            pair_idx += 1

    print("\n" + "─" * 80)
    print("3. DIFFERENT-PERSON PAIRWISE VERIFICATION (Expected: LOW similarity < 0.70)")
    print("─" * 80)
    print(f" {'Pair #':<8} | {'Identity A':<10} | {'Identity B':<10} | {'Norm A':<8} | {'Norm B':<8} | {'Dot Product':<12} | {'Cosine Sim':<11} | {'Status'}")
    print("─" * 80)

    pids = list(emb_data.keys())
    diff_scores = []
    diff_pair_idx = 1
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            item_a = emb_data[pids[i]][0]
            item_b = emb_data[pids[j]][0]
            dot = float(np.dot(item_a['unit_emb'], item_b['unit_emb']))
            cos_sim = dot / (np.linalg.norm(item_a['unit_emb']) * np.linalg.norm(item_b['unit_emb']))
            diff_scores.append(cos_sim)
            status = "SEPARATED (CORRECT)" if cos_sim < 0.72 else "COLLISION (FALSE MERGE)"
            print(f" {diff_pair_idx:<8d} | {pids[i]:<10} | {pids[j]:<10} | {item_a['norm']:<8.2f} | {item_b['norm']:<8.2f} | {dot:<12.4f} | {cos_sim:<11.4f} | {status}")
            diff_pair_idx += 1

    print("\n" + "─" * 80)
    print("4. SUMMARY MATHEMATICAL COMPARISON")
    print("─" * 80)
    print(f"  • Same-Person Average Cosine:       {np.mean(same_scores):.4f} (Range: [{np.min(same_scores):.4f} .. {np.max(same_scores):.4f}])")
    print(f"  • Different-Person Average Cosine:  {np.mean(diff_scores):.4f} (Range: [{np.min(diff_scores):.4f} .. {np.max(diff_scores):.4f}])")
    print(f"  • Discrimination Gap (Mean Diff):   {(np.mean(same_scores) - np.mean(diff_scores)):.4f}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    debug_reid_pairs()
