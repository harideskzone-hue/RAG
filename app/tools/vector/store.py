import os
import json
import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from app.platform.config.config import config
from pymilvus import connections, Collection, utility


from pydantic import BaseModel

class VectorMatch(BaseModel):
    id: str
    score: float
    camera_id: str
    timestamp: str
    description: str
    bbox: list[float] | None = None

class VectorStore(ABC):
    @abstractmethod
    async def search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None) -> list[VectorMatch]:
        pass

    @abstractmethod
    async def insert(self, collection_name: str, data: list[list[Any]]) -> None:
        pass

    @abstractmethod
    async def health(self) -> bool:
        pass


class NativeVectorStore(VectorStore):
    """NumPy-based local vector store for Docker-free execution."""
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dataset")
        self.data_dir = data_dir
        self.vectors_path = os.path.join(data_dir, "vectors.npy")
        self.meta_path = os.path.join(data_dir, "vector_metadata.json")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.vectors = np.array([])
        self.metadata = []
        self._load()

    def _load(self):
        if os.path.exists(self.vectors_path) and os.path.exists(self.meta_path):
            try:
                self.vectors = np.load(self.vectors_path)
                with open(self.meta_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Failed to load native vector store: {e}")

    def _save(self):
        # Atomic write for JSON metadata
        tmp_meta = self.meta_path + ".tmp"
        with open(tmp_meta, 'w') as f:
            json.dump(self.metadata, f)
        os.replace(tmp_meta, self.meta_path)
        # NumPy save
        np.save(self.vectors_path, self.vectors)

    async def search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None) -> list[VectorMatch]:
        return await asyncio.to_thread(self._search, collection_name, embedding, top_k, allowed_cameras)

    def _search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None) -> list[VectorMatch]:
        if len(self.vectors) == 0:
            return []
            
        query_vec = np.array(embedding)
        # Cosine similarity (1 - cosine distance)
        # A.B / (|A||B|)
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        # Handle zero norms
        norms[norms == 0] = 1e-10
        similarities = np.dot(self.vectors, query_vec) / norms
        
        # Sort indices by descending similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        matches = []
        for idx in top_indices:
            meta = self.metadata[idx]
            cam_id = meta.get("camera_id")
            
            # Camera RBAC enforcement
            if allowed_cameras is not None and cam_id not in allowed_cameras:
                continue
                
            similarity = max(0.0, float(similarities[idx]))
            matches.append(VectorMatch(
                id=meta.get("id"),
                score=similarity,
                camera_id=cam_id,
                timestamp=meta.get("timestamp"),
                description=meta.get("description", ""),
                bbox=meta.get("bbox")
            ))
            if len(matches) == top_k:
                break
        return matches

    async def insert(self, collection_name: str, data: list[list[Any]]) -> None:
        await asyncio.to_thread(self._insert, collection_name, data)
        
    def _insert(self, collection_name: str, data: list[list[Any]]) -> None:
        ids = data[0]
        embs = data[1]
        cams = data[2]
        times = data[3]
        descriptions = data[4] if len(data) > 4 else [""] * len(ids)
        bboxes = data[5] if len(data) > 5 else [None] * len(ids)
        
        new_vecs = np.array(embs)
        if len(self.vectors) == 0:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack((self.vectors, new_vecs))
            
        for i in range(len(ids)):
            self.metadata.append({
                "id": str(ids[i]),
                "camera_id": str(cams[i]),
                "timestamp": str(times[i]),
                "description": str(descriptions[i]) if descriptions[i] else "",
                "bbox": bboxes[i]
            })
            
        self._save()

    async def health(self) -> bool:
        return True


class MilvusVectorStore(VectorStore):
    def __init__(self, uri: str):
        self.uri = uri
        self._connected = False

    def _ensure_connection(self):
        if not self._connected:
            connections.connect("default", uri=self.uri)
            self._connected = True

    async def search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None) -> list[VectorMatch]:
        return await asyncio.to_thread(self._do_search, collection_name, embedding, top_k, allowed_cameras)

    def _do_search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None) -> list[VectorMatch]:
        self._ensure_connection()
        if not utility.has_collection(collection_name):
            return []
        col = Collection(collection_name)
        col.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Camera RBAC enforcement
        expr = None
        if allowed_cameras is not None:
            # Milvus expression syntax: camera_id in ["cam1", "cam2"]
            expr = f"camera_id in {allowed_cameras}"
            
        results = col.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["camera_id", "timestamp", "description", "bbox"]
        )
        matches = []
        for hits in results:
            for hit in hits:
                distance = hit.distance
                similarity = 1.0 / (1.0 + distance)
                matches.append(VectorMatch(
                    id=str(hit.id),
                    score=similarity,
                    camera_id=hit.entity.get("camera_id"),
                    timestamp=hit.entity.get("timestamp"),
                    description=hit.entity.get("description", ""),
                    bbox=hit.entity.get("bbox")
                ))
        return matches

    async def insert(self, collection_name: str, data: list[list[Any]]) -> None:
        await asyncio.to_thread(self._insert, collection_name, data)
        
    def _insert(self, collection_name: str, data: list[list[Any]]) -> None:
        self._ensure_connection()
        col = Collection(collection_name)
        col.insert(data)
        col.flush()

    async def health(self) -> bool:
        try:
            self._ensure_connection()
            return utility.has_collection("test_check")
        except Exception:
            return False


def get_vector_store() -> VectorStore:
    if config.mode == "native":
        return NativeVectorStore()
    return MilvusVectorStore(config.vector_backend_url)
