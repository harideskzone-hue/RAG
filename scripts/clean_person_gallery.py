#!/usr/bin/env python3
"""
clean_person_gallery.py — Quality Filtering & Deduplication for Person Gallery

Evaluates all crops stored in dataset/persons/ and:
1. Re-evaluates each crop using the strict CropQualitySelector.
2. Removes all blurry, dark, low-contrast, or occluded crops.
3. Removes duplicate/identical frame crops using perceptual cross-correlation.
4. Preserves only the top 3-5 pristine, high-resolution distinct keyframes per person.
5. Updates person.json metadata to match the cleaned gallery.
"""
import os
import sys
import json
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.cv.identity.quality import CropQualitySelector

def clean_gallery():
    persons_dir = PROJECT_ROOT / "dataset" / "persons"
    if not persons_dir.exists():
        print(f"Directory not found: {persons_dir}")
        return

    quality_selector = CropQualitySelector(
        min_width=48,
        min_height=96,
        min_area=4800,
        min_laplacian_var=60.0,
        min_head_laplacian_var=40.0,
        min_contrast_std=22.0,
        min_quality_score=0.50
    )

    total_persons = 0
    total_crops_scanned = 0
    total_crops_removed_quality = 0
    total_crops_removed_duplicate = 0
    total_crops_retained = 0
    removed_empty_persons = 0

    print("═" * 70)
    print("      VISTA AI — PERSON GALLERY QUALITY CLEANUP & DEDUPLICATION")
    print("═" * 70)

    person_dirs = sorted([d for d in persons_dir.iterdir() if d.is_dir() and d.name.startswith("PERSON_")])
    total_persons = len(person_dirs)
    print(f"▶ Scanning {total_persons} canonical person folders...\n")

    for pdir in person_dirs:
        crops_dir = pdir / "crops"
        if not crops_dir.exists():
            continue

        crop_files = sorted(list(crops_dir.glob("*.jpg")))
        if not crop_files:
            continue

        approved_scored_crops = []

        for cf in crop_files:
            total_crops_scanned += 1
            img = cv2.imread(str(cf))
            if img is None:
                cf.unlink(missing_ok=True)
                total_crops_removed_quality += 1
                continue

            q = quality_selector.assess_quality(img)
            score = float(q.get("score", 0))
            if not q.get("approved"):
                cf.unlink(missing_ok=True)
                total_crops_removed_quality += 1
            else:
                approved_scored_crops.append((score, img, cf))

        # Sort approved crops descending by quality score
        approved_scored_crops.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate approved crops
        distinct_crops = []
        for score, img, cf in approved_scored_crops:
            # Check if duplicate of already retained distinct crop
            is_dup = any(quality_selector.are_duplicate_crops(img, prev[1]) for prev in distinct_crops)
            if is_dup or len(distinct_crops) >= 5:
                # Remove duplicate or overflow crop
                cf.unlink(missing_ok=True)
                total_crops_removed_duplicate += 1
            else:
                distinct_crops.append((score, img, cf))
                total_crops_retained += 1

        # Update person.json metadata
        retained_ev_ids = [cf.stem for _, _, cf in distinct_crops]
        person_json_path = pdir / "person.json"
        if person_json_path.exists():
            try:
                with open(person_json_path, "r") as f:
                    person_data = json.load(f)
                person_data["evidence_ids"] = retained_ev_ids
                person_data["crop_count"] = len(retained_ev_ids)
                with open(person_json_path, "w") as f:
                    json.dump(person_data, f, indent=2)
            except Exception as err:
                print(f"  Warning updating {person_json_path}: {err}")

        # If no quality crops remain, remove folder if empty
        if not distinct_crops:
            try:
                import shutil
                shutil.rmtree(str(pdir))
                removed_empty_persons += 1
            except Exception:
                pass

    print("─" * 70)
    print("📊 Cleanup Results Summary:")
    print(f"  • Total Person Directories Processed:   {total_persons}")
    print(f"  • Total Crop Images Evaluated:           {total_crops_scanned}")
    print(f"  • Blurry/Low-Quality Crops Purged:       {total_crops_removed_quality}")
    print(f"  • Duplicate/Redundant Crops Purged:      {total_crops_removed_duplicate}")
    print(f"  • Pristine Distinct Quality Crops Kept:  {total_crops_retained}")
    if removed_empty_persons > 0:
        print(f"  • Empty/Zero-Quality Folders Removed:    {removed_empty_persons}")
    print("═" * 70 + "\n")

if __name__ == "__main__":
    clean_gallery()
