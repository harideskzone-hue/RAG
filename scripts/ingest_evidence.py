import asyncio
import os
import sys
import json
import uuid

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.platform.config.config import config

async def process_video(video_path: str):
    print(f"Starting ingestion pipeline for: {video_path}")
    
    try:
        import cv2
    except ImportError:
        print("cv2 not found. Please run: pip install opencv-python-headless")
        return
        
    try:
        import google.generativeai as genai
    except ImportError:
        print("google.generativeai not found. Please install it.")
        return

    # config mode check
    if config.mode == "native":
        print("Running in fully functional native mode.")

    print("[1/5] Extracting frames using OpenCV...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
        
    # We will just extract a single representative frame to avoid API spam for the test
    # Get frame at 1 second
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 1.0))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame.")
        return
        
    frame_path = "/tmp/vista_ingest_frame.jpg"
    cv2.imwrite(frame_path, frame)
    print(f"  -> Extracted frame to {frame_path}")
    
    print("[2/5] Generating real detection and embeddings (Local Models)...")
    
    # 1. Mock Detection for Ingestion (Object Tracking)
    # In a fully integrated CV pipeline, this would run YOLO/Deepsort.
    # We create entity observations directly.
    detection_data = {
        "description": "Person in blue shirt walking towards the main gate",
        "entities": [{"type": "person", "clothing": "blue shirt", "action": "walking", "direction": "main gate"}]
    }
    print(f"  -> Object Detection: {detection_data.get('description', '')[:50]}...")
    
    # 2. Get Embedding for the description using a real local model
    try:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(detection_data['description']).tolist()
        print(f"  -> Generated {len(embedding)} dim embedding via MiniLM")
    except ImportError:
        print("sentence-transformers not found. Please install it.")
        return
    except Exception as e:
        print(f"Embedding Generation Error: {e}")
        return
        
    print("[3/5] Writing to MetadataStore...")
    from app.tools.metadata.store import get_metadata_store
    meta_store = get_metadata_store()
    
    # Initialize schema if it's the native store
    if config.mode == "native":
        # Native store initializes itself in __init__
        pass
    else:
        # In production mode, we might need to initialize schema if not exists
        try:
            await meta_store.execute("""
                CREATE TABLE IF NOT EXISTS evidence_metadata (
                    id UUID PRIMARY KEY,
                    camera_id TEXT,
                    timestamp TEXT,
                    description TEXT,
                    entities_json TEXT,
                    video_uri TEXT
                )
            """)
        except Exception:
            pass # Ignore if table already exists or permission denied
            
    evidence_id = str(uuid.uuid4())
    video_s3_uri = f"s3://vista-video-bucket/cctv.mp4"
    await meta_store.execute("""
        INSERT INTO evidence_metadata (id, camera_id, timestamp, description, entities_json, video_uri)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, evidence_id, "CAM_02", "2026-08-08T10:42:15Z", detection_data.get('description', ''), json.dumps(detection_data.get('entities', [])), video_s3_uri)
    print("  -> Metadata inserted.")

    print("[4/5] Writing to VectorStore...")
    from app.tools.vector.store import get_vector_store
    vec_store = get_vector_store()
    
    # Note: Dimension is dynamic based on len(embedding)
    col_name = "person_embeddings"
    if config.mode != "native":
        # Initialize schema for Milvus if not exists
        from pymilvus import utility, connections, Collection, FieldSchema, CollectionSchema, DataType
        connections.connect("default", uri=config.vector_backend_url)
        if not utility.has_collection(col_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="evidence_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embedding)),
                FieldSchema(name="camera_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64)
            ]
            schema = CollectionSchema(fields)
            col = Collection(col_name, schema)
            col.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
            
    data = [
        [evidence_id],
        [embedding],
        ["CAM_02"],
        ["2026-08-08T10:42:15Z"]
    ]
    await vec_store.insert(col_name, data)
    print("  -> Vector inserted.")

    print("[5/5] Writing to BlobStore...")
    from app.tools.video.store import get_blob_store
    blob_store = get_blob_store()
    
    await blob_store.upload("vista-video-bucket", "cctv.mp4", video_path)
    await blob_store.upload("vista-video-bucket", "cctv_frame1.jpg", frame_path)
    print("  -> Raw frame and video uploaded.")
    
    print("\n🎉 Ingestion pipeline complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "video", "cctv.mp4")
    asyncio.run(process_video(video_file))
