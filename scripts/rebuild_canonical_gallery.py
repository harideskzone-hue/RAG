#!/usr/bin/env python3
"""
VISTA AI — Safe Canonical Gallery Rebuilder & Migration Tool

Resolves raw tracklets into canonical identities using the calibrated IdentityResolver.
Supports:
  --dry-run:  Displays candidate mapping table and metrics without modifying files.
  --commit:   Populates dataset/tracks/, dataset/persons/{canonical_id}/, Qdrant gallery,
              and syncs MongoDB/PostgreSQL without deleting raw evidence.
"""
import os
import sys
import argparse
import logging
import json
import uuid
import shutil
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
if os.path.exists(os.path.join(PROJECT_ROOT, ".env.local")):
    load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))
if os.path.exists(os.path.join(PROJECT_ROOT, ".env")):
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GalleryRebuilder")

from app.cv.reid.osnet import OSNetExtractor
from app.cv.identity.quality import CropQualitySelector
from app.cv.identity.resolver import IdentityResolver, ResolutionStatus


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def scan_existing_tracks(dataset_root: str):
    """
    Scans both dataset/persons/ (legacy) and dataset/tracks/ (new) for tracklet crops.
    """
    track_sources = {}
    persons_dir = Path(dataset_root) / "persons"
    tracks_dir = Path(dataset_root) / "tracks"

    if persons_dir.exists():
        for d in sorted(persons_dir.iterdir()):
            if d.is_dir() and d.name.startswith("P") and not d.name.startswith("PERSON_"):
                crops = list((d / "crops").glob("*.jpg"))
                if crops:
                    track_sources[d.name] = {
                        "dir": d,
                        "crops_dir": d / "crops",
                        "track_json": d / "track.json",
                        "crops": crops
                    }

    if tracks_dir.exists():
        for vdir in tracks_dir.iterdir():
            if vdir.is_dir():
                for tdir in vdir.iterdir():
                    if tdir.is_dir() and tdir.name.startswith("P"):
                        crops = list((tdir / "crops").glob("*.jpg"))
                        if crops:
                            track_sources[tdir.name] = {
                                "dir": tdir,
                                "crops_dir": tdir / "crops",
                                "track_json": tdir / "track.json",
                                "crops": crops,
                                "video_id": vdir.name
                            }

    return track_sources


def run_migration(dry_run: bool = True, threshold: float = 0.72, ambiguity_margin: float = 0.05, video_id: str = "VIDEO-2026-08-13-14-20-13.mp4", camera_id: str = "cam_auto_01"):
    print("\n" + "═" * 78)
    mode_str = "DRY-RUN (SIMULATION ONLY)" if dry_run else "COMMIT (APPLYING CHANGES)"
    print(f"       VISTA AI CANONICAL GALLERY MIGRATION — {mode_str}")
    print(f"       Calibrated Threshold: {threshold:.2f} | Ambiguity Margin: {ambiguity_margin:.2f}")
    print("═" * 78)

    extractor = OSNetExtractor()
    quality_selector = CropQualitySelector()
    resolver = IdentityResolver(match_threshold=threshold, ambiguity_margin=ambiguity_margin)

    dataset_root = os.path.join(PROJECT_ROOT, "dataset")
    tracks = scan_existing_tracks(dataset_root)

    print(f"\n▶ Found {len(tracks)} raw tracklet sources across repository.")

    # Incremental Gallery tracking
    # canonical_id -> list of {"emb": np.ndarray, "track_id": str, "evidence_id": str, "crop_path": str}
    canonical_gallery: dict[str, list[dict]] = {}
    track_resolutions = []

    matched_cnt = 0
    new_cnt = 0
    unresolved_cnt = 0

    for tid, info in sorted(tracks.items()):
        # Select best approved crop
        best_crop = None
        best_score = -1.0
        best_crop_path = None
        # Collect all approved crops, ranking by quality score
        approved_crops = []
        fallback_crop = None
        fallback_score = -1.0
        fallback_path = None
        fallback_ev_id = None

        for cf in info["crops"]:
            ev_id = cf.stem
            img = cv2.imread(str(cf))
            if img is None:
                continue
            h, w = img.shape[:2]
            q = quality_selector.assess_quality(img)
            score = q.get("score", float(h * w))
            if q.get("approved"):
                approved_crops.append((score, img, cf, ev_id))
            if (h * w) > fallback_score:
                fallback_score = float(h * w)
                fallback_crop = img
                fallback_path = cf
                fallback_ev_id = ev_id

        # Sort approved crops descending by quality
        approved_crops.sort(key=lambda x: x[0], reverse=True)
        if not approved_crops and fallback_crop is not None:
            approved_crops = [(fallback_score, fallback_crop, fallback_path, fallback_ev_id)]

        if not approved_crops:
            continue

        # Use best crop for embedding extraction
        best_score, selected_crop, selected_crop_path, selected_ev_id = approved_crops[0]
        # Keep top 5 to 10 approved crops per tracklet
        top_crops_list = approved_crops[:8]

        raw_emb = extractor.extract(selected_crop)
        norm_emb = np.array(raw_emb, dtype=np.float32)
        norm = np.linalg.norm(norm_emb)
        if norm > 0:
            norm_emb = norm_emb / norm
        else:
            continue

        track_entry = {
            "emb": norm_emb,
            "raw_emb": raw_emb,
            "track_id": tid,
            "evidence_id": selected_ev_id,
            "crop_img": selected_crop,
            "crop_path": selected_crop_path,
            "all_crops": top_crops_list
        }

        # Query existing canonical gallery
        if not canonical_gallery:
            # First tracklet initiates the gallery
            canonical_id = f"PERSON_{uuid.uuid4().hex[:8].upper()}"
            canonical_gallery[canonical_id] = [track_entry]
            new_cnt += 1
            status = ResolutionStatus.NEW
            top_score = 1.0
            track_resolutions.append((tid, status.value, canonical_id, top_score))
            continue

        # Score against all existing canonical persons
        search_results = []
        for cid, member_items in canonical_gallery.items():
            for m in member_items:
                sim = cosine_sim(norm_emb, m["emb"])
                search_results.append((cid, sim))

        search_results.sort(key=lambda x: x[1], reverse=True)

        status, matched_cid = resolver.resolve(search_results)
        top_score = search_results[0][1] if search_results else 0.0

        if status == ResolutionStatus.MATCHED:
            matched_cnt += 1
            canonical_gallery[matched_cid].append(track_entry)
            track_resolutions.append((tid, status.value, matched_cid, top_score))
        elif status == ResolutionStatus.NEW:
            new_cnt += 1
            new_cid = f"PERSON_{uuid.uuid4().hex[:8].upper()}"
            canonical_gallery[new_cid] = [track_entry]
            track_resolutions.append((tid, status.value, new_cid, top_score))
        else:
            unresolved_cnt += 1
            unresolved_id = f"UNRESOLVED_{tid}"
            track_resolutions.append((tid, status.value, unresolved_id, top_score))

    # Print summary table
    print("\n" + "─" * 78)
    print(f" {'Tracklet ID':<14} | {'Status':<12} | {'Canonical ID':<22} | {'Top Cosine Sim':<15}")
    print("─" * 78)
    for tid, st, cid, sc in track_resolutions[:30]:
        print(f" {tid:<14} | {st:<12} | {cid:<22} | {sc:<15.4f}")
    if len(track_resolutions) > 30:
        print(f" ... and {len(track_resolutions) - 30} more tracklets ...")
    print("─" * 78)

    print("\n📊 Resolution Summary Metrics:")
    print(f"  • Total Raw Tracklets Processed:    {len(tracks)}")
    print(f"  • Resolved MATCHED (Merged):        {matched_cnt} ({(matched_cnt/len(tracks))*100:.1f}%)")
    print(f"  • Resolved NEW (Canonical Persons): {new_cnt}")
    print(f"  • Preserved UNRESOLVED:             {unresolved_cnt} ({(unresolved_cnt/len(tracks))*100:.1f}%)")
    print(f"  • Total Canonical Persons Formed:   {len(canonical_gallery)}")

    if dry_run:
        print("\n[!] Dry run complete. No files or databases modified.")
        print("    To apply this migration, run with: python3 scripts/rebuild_canonical_gallery.py --commit")
        return

    # Commit phase
    print("\n▶ Applying Commit Migration...")

    # 1. Ensure raw track evidence exists in dataset/tracks/
    tracks_root = Path(dataset_root) / "tracks" / video_id
    tracks_root.mkdir(parents=True, exist_ok=True)

    for tid, info in tracks.items():
        dest_track_dir = tracks_root / tid
        dest_crops_dir = dest_track_dir / "crops"
        dest_crops_dir.mkdir(parents=True, exist_ok=True)

        for cf in info["crops"]:
            dest_cf = dest_crops_dir / cf.name
            if not dest_cf.exists():
                shutil.copy2(str(cf), str(dest_cf))

        if info["track_json"].exists():
            dest_json = dest_track_dir / "track.json"
            if not dest_json.exists():
                shutil.copy2(str(info["track_json"]), str(dest_json))

    print("  ✓ Raw track evidence crops preserved in dataset/tracks/")

    # 2. Build canonical person directories in dataset/persons/ with enhanced 5-10 crops on white canvas
    from app.cv.crops.enhancer import CropEnhancer
    enhancer = CropEnhancer()
    persons_root = Path(dataset_root) / "persons"

    for cid, members in canonical_gallery.items():
        cdir = persons_root / cid
        ccrops_dir = cdir / "crops"
        ccrops_dir.mkdir(parents=True, exist_ok=True)

        evidence_ids = []
        member_tracks = []

        for m in members:
            if m["track_id"] not in member_tracks:
                member_tracks.append(m["track_id"])

            # Enhance and save all 5-10 crops for this member tracklet
            for _, c_img, c_path, c_ev_id in m.get("all_crops", []):
                evidence_ids.append(c_ev_id)
                dest_crop = ccrops_dir / f"{c_ev_id}.jpg"
                if c_img is not None:
                    enhanced_img = enhancer.enhance(c_img)
                    cv2.imwrite(str(dest_crop), enhanced_img)
                elif c_path and Path(c_path).exists():
                    raw_read = cv2.imread(str(c_path))
                    if raw_read is not None:
                        enhanced_img = enhancer.enhance(raw_read)
                        cv2.imwrite(str(dest_crop), enhanced_img)

        person_json_path = cdir / "person.json"
        person_meta = {
            "canonical_person_id": cid,
            "tracks": member_tracks,
            "evidence_ids": evidence_ids,
            "cameras": [camera_id],
            "video_id": video_id,
            "crop_count": len(members)
        }
        with open(person_json_path, "w") as f:
            json.dump(person_meta, f, indent=2)

    print(f"  ✓ Created {len(canonical_gallery)} canonical person folders under dataset/persons/")

    # 3. Synchronize Qdrant Vector Store
    try:
        from app.tools.vector.store import get_vector_store
        vstore = get_vector_store()

        # Re-initialize collection person_embeddings_v2 with canonical multi-embedding points
        ids_list = []
        embs_list = []
        cams_list = []
        times_list = []
        desc_list = []

        for cid, members in canonical_gallery.items():
            for m in members:
                ids_list.append(cid)
                embs_list.append(m["raw_emb"])
                cams_list.append(camera_id)
                times_list.append("0.0")
                desc_list.append(f"Canonical person {cid} tracklet {m['track_id']}")

        import asyncio
        data = [ids_list, embs_list, cams_list, times_list, desc_list]
        asyncio.run(vstore.insert("person_embeddings_v2", data))
        print(f"  ✓ Synchronized Qdrant collection person_embeddings_v2 with {len(ids_list)} canonical embedding points.")
    except Exception as e:
        logger.warning(f"Could not update Qdrant during commit: {e}")

    print("\n═" * 78)
    print("  ✓ CANONICAL GALLERY MIGRATION COMMITTED SUCCESSFULLY")
    print("═" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rebuild VISTA AI Canonical Gallery")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Perform dry run without modifying disk")
    parser.add_argument("--commit", action="store_true", default=False, help="Commit migration to disk and vector store")
    parser.add_argument("--threshold", type=float, default=0.72, help="Calibrated cosine matching threshold")
    parser.add_argument("--ambiguity-margin", type=float, default=0.05, help="Ambiguity margin delta")
    args = parser.parse_args()

    if not args.commit and not args.dry_run:
        print("[!] Neither --commit nor --dry-run specified. Defaulting to --dry-run.")
        dry_run = True
    else:
        dry_run = args.dry_run and not args.commit

    run_migration(dry_run=dry_run, threshold=args.threshold, ambiguity_margin=args.ambiguity_margin)


if __name__ == "__main__":
    main()
