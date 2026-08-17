import os
import json
from pathlib import Path
import numpy as np
import torch
import cv2

from app.cv.reid.osnet import OSNetExtractor

def rebuild_store():
    meta_path = Path("dataset/metadata/VIDEO-2026-08-17-11-38-54.mp4.json")
    if not meta_path.exists():
        print("Metadata JSON not found!")
        return

    with open(meta_path) as f:
        video_meta = json.load(f)

    osnet = OSNetExtractor()

    vectors = []
    metadata = []

    for t in video_meta.get("tracks", []):
        tid = t.get("track_id")
        cpid = t.get("canonical_person_id") or f"P_{tid}"
        gender = t.get("gender", "individual")
        role = t.get("role", "customer")
        desc = t.get("description", "")
        behavior = t.get("behavior", "")
        loc = t.get("location", "")
        zone = t.get("spatial_zone", "")
        crop_url = t.get("crop_url", "")
        start_t = t.get("start_time_sec", 0.0)

        # Get best visual crop image
        crop_path = None
        if crop_url:
            local_p = Path(crop_url.lstrip("/media/"))
            if (Path("dataset") / local_p).exists():
                crop_path = Path("dataset") / local_p
        
        if not crop_path:
            tdir = Path(f"dataset/tracks/VIDEO-2026-08-17-11-38-54.mp4/{tid}/crops")
            if tdir.exists():
                jpgs = sorted(tdir.glob("*.jpg"), key=lambda f: f.stat().st_size, reverse=True)
                if jpgs:
                    crop_path = jpgs[0]

        if crop_path and crop_path.exists():
            img = cv2.imread(str(crop_path))
            if img is not None:
                emb = osnet.extract(img)
            else:
                emb = [0.0] * 512
        else:
            emb = [0.0] * 512

        vectors.append(emb)
        metadata.append({
            "id": cpid,
            "track_id": tid,
            "camera_id": "cam_auto_01",
            "timestamp": str(start_t),
            "description": desc,
            "bbox": [0, 0, 100, 100],
            "crop_url": crop_url,
            "attributes": {
                "gender": gender,
                "role": role,
                "behavior": behavior,
                "location": loc,
                "spatial_zone": zone,
                "crop_url": crop_url
            }
        })

    vectors_arr = np.array(vectors, dtype=np.float32)

    # Save to native vector store files
    np.save("dataset/vectors_person_embeddings_v2.npy", vectors_arr)
    with open("dataset/meta_person_embeddings_v2.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Rebuilt native vector store with {len(metadata)} clean OSNet embeddings.")

if __name__ == "__main__":
    rebuild_store()
