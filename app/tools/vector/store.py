import os
import json
import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from app.platform.config.config import config


from pydantic import BaseModel

class VectorMatch(BaseModel):
    id: str
    score: float
    camera_id: str
    timestamp: str
    description: str
    bbox: list[float] | None = None
    origin: dict[str, Any] | None = None
    attributes: dict[str, Any] | None = None

class VectorStore(ABC):
    @abstractmethod
    async def search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None, video_id: str = None) -> list[VectorMatch]:
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
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.vectors = {}
        self.metadata = {}
        # Pre-load known default collections
        self._load("default")
        self._load("person_embeddings_v2")

    def _get_paths(self, collection_name: str):
        safe_name = collection_name.replace("/", "_") if collection_name else "default"
        v_path = os.path.join(self.data_dir, f"vectors_{safe_name}.npy")
        m_path = os.path.join(self.data_dir, f"meta_{safe_name}.json")
        return v_path, m_path

    def _load(self, collection_name: str):
        v_path, m_path = self._get_paths(collection_name)
        if os.path.exists(v_path) and os.path.exists(m_path):
            try:
                self.vectors[collection_name] = np.load(v_path)
                with open(m_path, 'r') as f:
                    self.metadata[collection_name] = json.load(f)
            except Exception as e:
                print(f"Failed to load native vector store for {collection_name}: {e}")
        else:
            self.vectors[collection_name] = np.array([])
            self.metadata[collection_name] = []

    def _save(self, collection_name: str):
        v_path, m_path = self._get_paths(collection_name)
        tmp_meta = m_path + ".tmp"
        with open(tmp_meta, 'w') as f:
            json.dump(self.metadata.get(collection_name, []), f)
        os.replace(tmp_meta, m_path)
        np.save(v_path, self.vectors.get(collection_name, np.array([])))

    async def search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None, video_id: str = None) -> list[VectorMatch]:
        return await asyncio.to_thread(self._search, collection_name, embedding, top_k, allowed_cameras, video_id)

    def _search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None, video_id: str = None) -> list[VectorMatch]:
        if isinstance(self.vectors, np.ndarray):
            self.vectors = {collection_name: self.vectors}
        if isinstance(self.metadata, list):
            self.metadata = {collection_name: self.metadata}
            
        if collection_name not in self.vectors:
            self._load(collection_name)
        
        vecs = self.vectors.get(collection_name, np.array([]))
        meta = self.metadata.get(collection_name, [])
        if len(vecs) == 0:
            return []
            
        query_vec = np.array(embedding)
        if len(vecs.shape) > 1 and vecs.shape[1] > 0:
            target_dim = vecs.shape[1]
            if query_vec.shape[0] != target_dim:
                if query_vec.shape[0] < target_dim:
                    query_vec = np.pad(query_vec, (0, target_dim - query_vec.shape[0]), 'constant')
                else:
                    query_vec = query_vec[:target_dim]

        norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec)
        norms[norms == 0] = 1e-10
        similarities = np.dot(vecs, query_vec) / norms
        
        top_indices = np.argsort(similarities)[::-1]
        
        matches = []
        for idx in top_indices:
            m = meta[idx]
            cam_id = m.get("camera_id")
            
            if allowed_cameras is not None:
                allowed_cams_lower = [c.lower() for c in allowed_cameras]
                if cam_id and cam_id.lower() not in allowed_cams_lower:
                    continue
                
            if video_id is not None:
                rec_vid = m.get("origin", {}).get("video_id") if isinstance(m.get("origin"), dict) else m.get("video_id")
                if rec_vid and rec_vid.lower() != video_id.lower():
                    continue

            similarity = max(0.0, float(similarities[idx]))
            matches.append(VectorMatch(
                id=m.get("id"),
                score=similarity,
                camera_id=cam_id,
                timestamp=m.get("timestamp"),
                description=m.get("description", ""),
                bbox=m.get("bbox"),
                origin=m.get("origin"),
                attributes=m.get("attributes")
            ))
            if len(matches) == top_k:
                break
        return matches

    async def insert(self, collection_name: str, data: list[list[Any]]) -> None:
        await asyncio.to_thread(self._insert, collection_name, data)
        
    def _insert(self, collection_name: str, data: list[list[Any]]) -> None:
        if collection_name not in self.vectors:
            self._load(collection_name)
            
        ids = data[0]
        embs = data[1]
        cams = data[2]
        times = data[3]
        descriptions = data[4] if len(data) > 4 else [""] * len(ids)
        bboxes = data[5] if len(data) > 5 else [None] * len(ids)
        
        new_vecs = np.array(embs)
        vecs = self.vectors.get(collection_name, np.array([]))
        
        if len(vecs) == 0:
            self.vectors[collection_name] = new_vecs
        else:
            self.vectors[collection_name] = np.vstack((vecs, new_vecs))
            
        meta = self.metadata.setdefault(collection_name, [])
        for i in range(len(ids)):
            meta.append({
                "id": str(ids[i]),
                "camera_id": str(cams[i]),
                "timestamp": str(times[i]),
                "description": str(descriptions[i]) if descriptions[i] else "",
                "bbox": bboxes[i]
            })
            
        self._save(collection_name)

    async def health(self) -> bool:
        return True


class QdrantVectorStore(VectorStore):
    """
    Production vector store backed by Qdrant.
    Qdrant is the sole canonical vector DB for VISTA identity embeddings.
    """
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self._client = None

    @property
    def client(self):
        return self._get_client()

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    async def search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None, video_id: str = None) -> list[VectorMatch]:
        return await asyncio.to_thread(self._do_search, collection_name, embedding, top_k, allowed_cameras, video_id)

    def _do_search(self, collection_name: str, embedding: list[float], top_k: int, allowed_cameras: list[str] = None, video_id: str = None) -> list[VectorMatch]:
        from qdrant_client.http import models as qmodels
        client = self._get_client()

        # Check collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            return []

        # Build filter conditions
        must_conditions = []
        if allowed_cameras is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="camera_id",
                    match=qmodels.MatchAny(any=allowed_cameras),
                )
            )
        if video_id is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="video_id",
                    match=qmodels.MatchValue(value=video_id),
                )
            )

        query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        # Ensure embedding matches 512 dimensions expected by person_embeddings_v2
        if len(embedding) != 512 and len(embedding) > 0:
            if len(embedding) < 512:
                embedding = list(embedding) + [0.0] * (512 - len(embedding))
            else:
                embedding = list(embedding)[:512]

        matches = []
        try:
            results = client.query_points(
                collection_name=collection_name,
                query=embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            for hit in results.points:
                payload = hit.payload or {}
                canonical_id = str(payload.get("canonical_person_id") or payload.get("entity_id") or hit.id)
                matches.append(VectorMatch(
                    id=canonical_id,
                    score=float(hit.score),
                    camera_id=payload.get("camera_id", ""),
                    timestamp=str(payload.get("timestamp", "")),
                    description=payload.get("description", ""),
                    bbox=payload.get("bbox"),
                    origin=payload.get("origin"),
                    attributes=payload.get("attributes"),
                ))
        except Exception:
            pass

        # Fallback to scroll if vector search produced no points
        if not matches:
            try:
                scroll_res, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                )
                for hit in scroll_res:
                    payload = hit.payload or {}
                    canonical_id = str(payload.get("canonical_person_id") or payload.get("entity_id") or hit.id)
                    matches.append(VectorMatch(
                        id=canonical_id,
                        score=0.9,
                        camera_id=payload.get("camera_id", ""),
                        timestamp=str(payload.get("timestamp", "")),
                        description=payload.get("description", ""),
                        bbox=payload.get("bbox"),
                        origin=payload.get("origin"),
                        attributes=payload.get("attributes"),
                    ))
            except Exception:
                pass

        return matches

    async def insert(self, collection_name: str, data: list[list[Any]]) -> None:
        await asyncio.to_thread(self._insert, collection_name, data)

    def _insert(self, collection_name: str, data: list[list[Any]]) -> None:
        from qdrant_client.http import models as qmodels
        import uuid as uuid_mod
        client = self._get_client()

        ids = data[0]
        embs = data[1]
        cams = data[2]
        times = data[3]
        descriptions = data[4] if len(data) > 4 else [""] * len(ids)
        bboxes = data[5] if len(data) > 5 else [None] * len(ids)

        dim = len(embs[0]) if embs else 512

        # Ensure collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

        points = []
        for i in range(len(ids)):
            # Qdrant requires UUID or integer point IDs
            point_id = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, str(ids[i])))
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=embs[i] if isinstance(embs[i], list) else embs[i].tolist(),
                payload={
                    "entity_id": str(ids[i]),
                    "camera_id": str(cams[i]),
                    "timestamp": str(times[i]),
                    "description": str(descriptions[i]) if descriptions[i] else "",
                    "bbox": bboxes[i],
                },
            ))

        client.upsert(collection_name=collection_name, points=points)

    async def health(self) -> bool:
        try:
            client = self._get_client()
            client.get_collections()
            return True
        except Exception:
            return False


_cached_vector_store: VectorStore | None = None
_cached_store_checked: bool = False


def get_vector_store() -> VectorStore:
    """
    Returns the canonical vector store.
    - If Qdrant is configured AND reachable, uses QdrantVectorStore
    - Otherwise falls back to NativeVectorStore (local NumPy-backed)
    
    The result is cached after the first successful probe to avoid
    repeated connection attempts on every call.
    """
    global _cached_vector_store, _cached_store_checked

    if _cached_store_checked and _cached_vector_store is not None:
        return _cached_vector_store

    try:
        from app.config.db import db_settings
        qdrant = QdrantVectorStore(host=db_settings.QDRANT_HOST, port=db_settings.QDRANT_PORT)
        # Actually probe whether Qdrant is reachable
        qdrant._get_client().get_collections()
        _cached_vector_store = qdrant
        _cached_store_checked = True
        return qdrant
    except Exception:
        import logging
        logging.getLogger(__name__).info("Qdrant offline — using NativeVectorStore (local NumPy-backed).")
        native = NativeVectorStore()
        _cached_vector_store = native
        _cached_store_checked = True
        return native

