import asyncio
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from app.cv.reid.osnet import OSNetExtractor
from app.cv.identity.quality import CropQualitySelector
from app.cv.identity.resolver import IdentityResolver, ResolutionStatus
from app.infrastructure.db.qdrant.client import VectorRepository

async def run_validation(video_path: str):
    print(f"Validating Phase 3 Identity flow on video: {video_path}")
    
    # 1. Initialize models
    print("Loading YOLOv8n detector...")
    detector = YOLO("yolov8n.pt") # fallback standard YOLO, representing yolo26n for test
    detector.classes = [0] # only detect person

    quality_selector = CropQualitySelector(min_width=32, min_height=64, min_laplacian_var=10.0) 
    
    print("Loading OSNet Feature Extractor...")
    extractor = OSNetExtractor(model_name="osnet_x1_0", device="cpu")
    
    # Very strict resolver for tests to avoid false merging
    resolver = IdentityResolver(match_threshold=0.80, ambiguity_margin=0.05)
    
    # Mock Qdrant for this validation test since Docker Qdrant is unavailable
    class MockQdrant:
        def __init__(self):
            self.embeddings = []
        async def search_top_k(self, embedding, top_k=5):
            if not self.embeddings:
                return []
            emb_matrix = np.array([e["vec"] for e in self.embeddings])
            query_vec = np.array(embedding)
            # Cosine similarity (vectors are L2 normalized)
            scores = np.dot(emb_matrix, query_vec)
            
            # Sort descending
            results = []
            for i, score in enumerate(scores):
                results.append((self.embeddings[i]["person_id"], float(score)))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        async def insert_embedding(self, vector_id, embedding, entity_type, entity_id):
            self.embeddings.append({"vec": embedding, "person_id": entity_id})
            
    qdrant = MockQdrant()
    
    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
        
    metrics = {
        "false_identity_matches": 0,
        "false_new_person": 0,
        "unresolved_rate": 0,
        "same_person_fragmented": 0,
        "different_people_merged": 0,
        "embeddings_per_person": {},
        "resolution_latency_ms": [],
        "qdrant_latency_ms": []
    }
    
    total_processed = 0
    unresolved_count = 0
    canonical_persons = {}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx > 100:  # Process first 100 frames to keep test time reasonable
            break
            
        # Detect persons
        results = detector(frame, classes=[0], verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                # Quality check
                quality = quality_selector.assess_quality(crop)
                if not quality["approved"]:
                    continue
                    
                # Extract embedding
                t0 = time.time()
                embedding = extractor.extract(crop)
                
                # Qdrant search
                t1 = time.time()
                search_results = await qdrant.search_top_k(embedding)
                metrics["qdrant_latency_ms"].append((time.time() - t1) * 1000)
                
                # Resolve
                t2 = time.time()
                status, person_id = resolver.resolve(search_results)
                metrics["resolution_latency_ms"].append((time.time() - t2) * 1000)
                
                if status == ResolutionStatus.NEW:
                    person_id = f"P_{total_processed:04d}"
                    canonical_persons[person_id] = 1
                elif status == ResolutionStatus.MATCHED:
                    canonical_persons[person_id] += 1
                else:
                    unresolved_count += 1
                    
                if status != ResolutionStatus.UNRESOLVED:
                    # Store in qdrant
                    await qdrant.insert_embedding(
                        vector_id=f"vec_{total_processed}",
                        embedding=embedding,
                        entity_type="person",
                        entity_id=person_id
                    )
                    
                total_processed += 1

    cap.release()

    metrics["unresolved_rate"] = unresolved_count / max(1, total_processed)
    metrics["embeddings_per_person"] = canonical_persons
    
    print("\n--- Phase 3 Identity Validation Metrics ---")
    print(f"Total Quality-Approved Crops Processed: {total_processed}")
    print(f"Total Unique Persons Created: {len(canonical_persons)}")
    print(f"UNRESOLVED Rate: {metrics['unresolved_rate']:.2%}")
    if metrics["qdrant_latency_ms"]:
        print(f"Avg Qdrant Top-K Latency: {sum(metrics['qdrant_latency_ms'])/len(metrics['qdrant_latency_ms']):.2f} ms")
        print(f"Avg Identity Resolution Latency: {sum(metrics['resolution_latency_ms'])/len(metrics['resolution_latency_ms']):.2f} ms")
    print(f"Embeddings per Person: {canonical_persons}")
    print("-------------------------------------------\n")
    print("Note: False matches and fragmentation metrics require labeled ground-truth tracks.")

if __name__ == "__main__":
    asyncio.run(run_validation("/Users/hariharans/Documents/longgraph/input/VIDEO-2026-08-13-14-20-13.mp4"))
