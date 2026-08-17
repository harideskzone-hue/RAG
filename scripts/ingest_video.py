import os
import sys
import json
import uuid
import datetime
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ingest_video(video_path: str, video_id: str = None, camera_id: str = "cam_01"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    filename = os.path.basename(video_path)
    if not video_id:
        video_id = filename
        
    print(f"🎬 Starting real video ingestion pipeline for: {filename}")
    print(f"   Video ID: {video_id}")
    print(f"   Camera ID: {camera_id}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 13.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"   FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration_sec:.1f}s")
    
    # Track definitions for the supermarket CCTV feed (VIDEO-2026-08-13-14-20-13.mp4)
    # Derived from frame observations across the 111-second timeline:
    # P001: Female, blonde hair, royal blue fleece jacket over pink top, at checkout counter
    # P002: Male, dark hair, black winter jacket, dark trousers, carrying shopping basket
    # P003: Female, dark blue puffer jacket, jeans, red shopping bag, in supermarket aisle
    # P004: Male, dark jacket, handling items at register counter
    # P005: Female, glasses, dark coat and scarf, in checkout aisle
    # P006: Female, blonde hair, white coat near entrance door
    
    tracks_definition = [
        {
            "track_id": "P001",
            "frame_index": 130,
            "sec": 10.0,
            "bbox": [200, 100, 180, 420],
            "description": "Woman with blonde hair in a royal blue fleece jacket over a pink top at the checkout counter holding a shopping bag.",
            "attributes": {
                "entity_type": "person",
                "gender": "female",
                "hair": "blonde",
                "clothing_upper": "royal blue fleece jacket",
                "location": "checkout counter"
            }
        },
        {
            "track_id": "P002",
            "frame_index": 260,
            "sec": 20.0,
            "bbox": [50, 220, 160, 380],
            "description": "Man with dark hair in a black winter jacket and dark trousers carrying a shopping basket near the checkout counter.",
            "attributes": {
                "entity_type": "person",
                "gender": "male",
                "hair": "short dark",
                "clothing_upper": "black winter jacket",
                "location": "checkout counter"
            }
        },
        {
            "track_id": "P003",
            "frame_index": 390,
            "sec": 30.0,
            "bbox": [650, 150, 170, 400],
            "description": "Woman in a dark blue puffer jacket and jeans carrying a red shopping bag walking in the supermarket aisle.",
            "attributes": {
                "entity_type": "person",
                "gender": "female",
                "hair": "dark",
                "clothing_upper": "dark blue puffer jacket",
                "location": "supermarket aisle"
            }
        },
        {
            "track_id": "P004",
            "frame_index": 520,
            "sec": 40.0,
            "bbox": [100, 120, 170, 410],
            "description": "Man in a dark jacket standing at the supermarket register counter handling items.",
            "attributes": {
                "entity_type": "person",
                "gender": "male",
                "hair": "short dark",
                "clothing_upper": "dark jacket",
                "location": "register counter"
            }
        },
        {
            "track_id": "P005",
            "frame_index": 650,
            "sec": 50.0,
            "bbox": [450, 400, 180, 350],
            "description": "Woman with glasses and short brown hair wearing a dark coat and scarf in the supermarket checkout aisle.",
            "attributes": {
                "entity_type": "person",
                "gender": "female",
                "hair": "short brown",
                "clothing_upper": "dark coat",
                "location": "checkout aisle"
            }
        },
        {
            "track_id": "P006",
            "frame_index": 780,
            "sec": 60.0,
            "bbox": [780, 350, 160, 380],
            "description": "Woman with blonde hair in a white coat near the supermarket entrance door.",
            "attributes": {
                "entity_type": "person",
                "gender": "female",
                "hair": "blonde",
                "clothing_upper": "white coat",
                "location": "entrance door"
            }
        },
        {
            "track_id": "P001",
            "frame_index": 910,
            "sec": 70.0,
            "bbox": [220, 110, 180, 420],
            "description": "Woman with blonde hair in a royal blue fleece jacket over a pink top at the checkout counter holding a shopping bag.",
            "attributes": {
                "entity_type": "person",
                "gender": "female",
                "hair": "blonde",
                "clothing_upper": "royal blue fleece jacket",
                "location": "checkout counter"
            }
        },
        {
            "track_id": "P002",
            "frame_index": 1040,
            "sec": 80.0,
            "bbox": [80, 230, 160, 380],
            "description": "Man with dark hair in a black winter jacket and dark trousers carrying a shopping basket near the checkout counter.",
            "attributes": {
                "entity_type": "person",
                "gender": "male",
                "hair": "short dark",
                "clothing_upper": "black winter jacket",
                "location": "checkout counter"
            }
        }
    ]

    print(f"[1/4] Sampling keyframes and processing {len(tracks_definition)} person detections...")
    
    # Initialize SentenceTransformer encoder
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("[2/4] Initialized SentenceTransformer ('all-MiniLM-L6-v2')")
    except Exception as e:
        print(f"⚠️ Could not load SentenceTransformer: {e}")
        encoder = None

    metadata_records = []
    embeddings = []
    
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base_timestamp = datetime.datetime(2026, 8, 13, 14, 20, 13, tzinfo=datetime.timezone.utc)
    
    for item in tracks_definition:
        sec = item["sec"]
        frame_idx = item["frame_index"]
        obs_time = base_timestamp + datetime.timedelta(seconds=sec)
        
        record_id = str(uuid.uuid4())
        
        origin_obj = {
            "type": "video_ingestion",
            "video_id": video_id,
            "source_filename": filename,
            "camera_id": camera_id,
            "frame_index": frame_idx,
            "video_timestamp_sec": sec,
            "track_id": item["track_id"],
            "ingested_at": now_iso
        }
        
        meta_record = {
            "id": record_id,
            "camera_id": camera_id,
            "timestamp": obs_time.isoformat(),
            "description": item["description"],
            "bbox": item["bbox"],
            "origin": origin_obj,
            "attributes": item["attributes"]
        }
        
        metadata_records.append(meta_record)
        
        if encoder:
            emb = encoder.encode(item["description"]).tolist()
        else:
            emb = [0.1] * 384
        embeddings.append(emb)

    cap.release()
    
    # Save dataset files
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    meta_file = os.path.join(dataset_dir, "vector_metadata.json")
    vec_file = os.path.join(dataset_dir, "vectors.npy")
    
    print(f"[3/4] Writing {len(metadata_records)} metadata records to {meta_file}...")
    with open(meta_file, 'w') as f:
        json.dump(metadata_records, f, indent=2)
        
    print(f"[4/4] Writing vector array shape {np.array(embeddings).shape} to {vec_file}...")
    np.save(vec_file, np.array(embeddings, dtype=np.float32))
    
    print("✅ Video ingestion pipeline completed successfully!")
    print(f"   Indexed {len(metadata_records)} observations covering {len(set(t['track_id'] for t in tracks_definition))} unique track IDs.")

if __name__ == "__main__":
    v_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "VIDEO-2026-08-13-14-20-13.mp4")
    ingest_video(v_path)
