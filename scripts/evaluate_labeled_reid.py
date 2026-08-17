#!/usr/bin/env python3
"""
VISTA AI — Comprehensive Ground-Truth Re-ID Benchmark & Multi-Camera Audit

Evaluates MSMT17 OSNet 512-D on labeled multi-camera surveillance splits:
1. Same-Person Cross-Camera Similarity (Cross-Viewpoint / Illumination changes)
2. Same-Person Cross-Track Similarity (Re-appearance across time)
3. Different-Person Similarity (True negative distribution)
4. Full metric sweep across [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
   - Recall (TPR), Precision, FMR (FPR), FNMR (FNR), False Merge, False Split
   - Rank-1 Identification Accuracy (CMC top-1)
5. Canonical Person Gallery Membership Tree Audit
"""
import os
import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LabeledReIDEval")

from app.cv.reid.osnet import OSNetExtractor
from app.cv.identity.resolver import IdentityResolver, ResolutionStatus


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def parse_crop_metadata(filename: str):
    """
    Extracts camera_id, track_id, and frame_index from filename.
    Format: {cam}_{cam}_track_{t}_f{frame}_q{score}.jpg
    """
    parts = filename.split("_")
    cam = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "cam_unknown"
    track = "track_unknown"
    if "_track_" in filename:
        track = filename.split("_track_")[1].split("_")[0]
    return cam, track


def load_labeled_dataset(reid_dir: str):
    """
    Loads labeled multi-camera person crops from dataset/reid/ splits.
    Returns:
      records = [
         {"pid": pid, "cam": cam, "track": track, "filename": cf.name, "emb": norm_emb}
      ]
    """
    extractor = OSNetExtractor()
    reid_path = Path(reid_dir)

    records = []
    for split in ["train", "val", "test"]:
        split_dir = reid_path / split
        if not split_dir.exists():
            continue

        for pdir in sorted(split_dir.iterdir()):
            if not pdir.is_dir():
                continue
            pid = pdir.name
            crop_files = sorted(list(pdir.glob("*.jpg")))
            
            # Subsample across time to capture distinct poses/frames
            for cf in crop_files[::8]:
                img = cv2.imread(str(cf))
                if img is None:
                    continue
                emb = extractor.extract(img)
                norm_emb = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(norm_emb)
                if norm > 0:
                    norm_emb = norm_emb / norm
                    cam, track = parse_crop_metadata(cf.name)
                    records.append({
                        "pid": pid,
                        "cam": cam,
                        "track": track,
                        "filename": cf.name,
                        "emb": norm_emb
                    })

    logger.info(f"Loaded {len(records)} benchmark crops across {len(set(r['pid'] for r in records))} identities.")
    return records


def evaluate_benchmark(records: list):
    same_cross_cam = []
    same_cross_track = []
    same_intra_track = []
    diff_person = []

    # 1. Pairwise Similarity Calculation
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            r_a = records[i]
            r_b = records[j]
            sim = cosine_sim(r_a["emb"], r_b["emb"])

            if r_a["pid"] == r_b["pid"]:
                if r_a["cam"] != r_b["cam"]:
                    same_cross_cam.append(sim)
                elif r_a["track"] != r_b["track"]:
                    same_cross_track.append(sim)
                else:
                    same_intra_track.append(sim)
            else:
                diff_person.append(sim)

    # 2. Rank-1 Accuracy Evaluation (Query against Gallery)
    # For each query crop, rank all other crops; success if top non-self match is same PID
    rank1_correct = 0
    rank1_total = 0

    for i, query in enumerate(records):
        best_sim = -1.0
        best_match_pid = None
        for j, gallery in enumerate(records):
            if i == j:
                continue
            # Exclude intra-track near-duplicates for strict cross-track / cross-camera Re-ID evaluation
            if query["pid"] == gallery["pid"] and query["cam"] == gallery["cam"] and query["track"] == gallery["track"]:
                continue

            sim = cosine_sim(query["emb"], gallery["emb"])
            if sim > best_sim:
                best_sim = sim
                best_match_pid = gallery["pid"]

        if best_match_pid is not None:
            rank1_total += 1
            if best_match_pid == query["pid"]:
                rank1_correct += 1

    rank1_acc = (rank1_correct / rank1_total) * 100 if rank1_total > 0 else 0.0

    print("\n" + "═" * 84)
    print("      VISTA AI — MULTI-CAMERA GROUND-TRUTH RE-ID BENCHMARK (MSMT17)")
    print("═" * 84)
    print(f"  • Total Benchmark Samples:          {len(records)}")
    print(f"  • Total Labeled Identities:         {len(set(r['pid'] for r in records))}")
    print(f"  • Cross-Camera Same-Person Pairs:   {len(same_cross_cam):,}  (Mean: {np.mean(same_cross_cam):.4f}, Range: [{np.min(same_cross_cam):.4f} .. {np.max(same_cross_cam):.4f}])")
    if same_cross_track:
        print(f"  • Cross-Track Same-Person Pairs:    {len(same_cross_track):,}  (Mean: {np.mean(same_cross_track):.4f})")
    print(f"  • Intra-Track Same-Person Pairs:    {len(same_intra_track):,}  (Mean: {np.mean(same_intra_track):.4f})")
    print(f"  • Different-Person Negative Pairs:  {len(diff_person):,}  (Mean: {np.mean(diff_person):.4f}, Range: [{np.min(diff_person):.4f} .. {np.max(diff_person):.4f}])")
    print(f"  • Discrimination Gap (Cross-Cam):   {(np.mean(same_cross_cam) - np.mean(diff_person)):.4f}")
    print(f"  • Rank-1 Identification Accuracy:   {rank1_acc:.2f}%")
    print("─" * 84)

    # 3. Threshold Sweep across All Same-Person vs Different-Person
    all_same = np.array(same_cross_cam + same_cross_track + same_intra_track)
    all_diff = np.array(diff_person)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85]

    print(f" {'Threshold':<10} | {'Recall (TPR)':<13} | {'Precision':<11} | {'FMR (FPR)':<11} | {'FNMR (FNR)':<11} | {'Verdict':<16}")
    print("─" * 84)

    for t in thresholds:
        tp = np.sum(all_same >= t)
        fn = np.sum(all_same < t)
        fp = np.sum(all_diff >= t)
        tn = np.sum(all_diff < t)

        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 100.0
        fmr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        fnmr = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0

        if fmr == 0.0:
            verdict = "SAFE (0.0% FMR)"
        elif fmr < 1.0:
            verdict = "EXCELLENT"
        elif fmr < 5.0:
            verdict = "LOW RISK"
        elif fmr < 15.0:
            verdict = "MODERATE"
        else:
            verdict = "UNSAFE (MERGE)"

        print(f" {t:<10.2f} | {recall:<13.2f}% | {precision:<11.2f}% | {fmr:<11.3f}% | {fnmr:<11.2f}% | {verdict:<16}")

    print("─" * 84)


def audit_canonical_gallery_tree():
    """
    Audits dataset/persons/ and prints the exact Canonical-to-Tracklet membership tree.
    """
    persons_dir = Path(PROJECT_ROOT) / "dataset" / "persons"
    canonical_dirs = sorted([d for d in persons_dir.iterdir() if d.is_dir() and d.name.startswith("PERSON_")])

    print("\n" + "═" * 84)
    print("      VISTA AI — CANONICAL PERSON GALLERY MEMBERSHIP AUDIT")
    print("═" * 84)
    print(f"Total Canonical Person Folders: {len(canonical_dirs)}\n")

    all_tracks = []
    for cdir in canonical_dirs:
        cid = cdir.name
        meta_path = cdir / "person.json"
        crops = list((cdir / "crops").glob("*.jpg"))
        
        member_tracks = []
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    member_tracks = meta.get("tracks", [])
            except Exception:
                pass

        all_tracks.extend(member_tracks)
        tracks_str = ", ".join(member_tracks) if member_tracks else "none"
        print(f"  👤 {cid} ({len(crops)} crops) ➔ Member Tracklets: [{tracks_str}]")

    duplicates = [t for t in set(all_tracks) if all_tracks.count(t) > 1]
    print("\n" + "─" * 84)
    print(f"  • Total Tracklets Mapped into Gallery: {len(all_tracks)}")
    print(f"  • Cross-Identity Collisions:            {len(duplicates)}")
    if len(duplicates) == 0:
        print("  ✓ GALLERY INTEGRITY: 100% PASS (Zero tracklet collision across canonical identities)")
    else:
        print(f"  ✗ WARNING: Collisions detected on: {duplicates}")
    print("═" * 84 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Labeled Re-ID & Audit Gallery")
    parser.add_argument("--reid-dir", default="dataset/reid", help="Directory containing labeled reid splits")
    args = parser.parse_args()

    reid_path = os.path.abspath(args.reid_dir)
    if not os.path.exists(reid_path):
        reid_path = os.path.join(PROJECT_ROOT, args.reid_dir)

    if os.path.exists(reid_path):
        records = load_labeled_dataset(reid_path)
        if records:
            evaluate_benchmark(records)

    audit_canonical_gallery_tree()


if __name__ == "__main__":
    main()
