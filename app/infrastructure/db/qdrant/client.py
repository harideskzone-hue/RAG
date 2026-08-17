from typing import Any, List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.config.db import db_settings

class VectorRepository:
    """
    Generic VectorRepository for all embedding types.
    Prepares infrastructure for Phase 3 embeddings.
    """
    def __init__(self):
        # We will connect to local Qdrant for Phase 2 infrastructure checks.
        self.client = AsyncQdrantClient(host=db_settings.QDRANT_HOST, port=db_settings.QDRANT_PORT)
        self.collection_name = "vista_embeddings"
        self.vector_size = 512  # Arbitrary for now, to be tuned in Phase 3 (e.g. CLIP/OSNet)

    async def setup_collection(self):
        """
        Create the collection if it doesn't exist.
        """
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            )
            # Create payload indexes for filtering
            await self.client.create_payload_index(self.collection_name, "entity_type", "keyword")
            await self.client.create_payload_index(self.collection_name, "video_id", "keyword")
            await self.client.create_payload_index(self.collection_name, "track_id", "keyword")

    async def insert_embedding(
        self,
        vector_id: str,
        embedding: List[float],
        entity_type: str,
        entity_id: str,
        video_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        track_id: Optional[str] = None,
        embedding_type: str = "visual",
    ) -> None:
        """
        Insert a generic vector. 
        Note: In Phase 2, this is only used with synthetic test vectors.
        """
        payload = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "video_id": video_id,
            "camera_id": camera_id,
            "track_id": track_id,
            "embedding_type": embedding_type,
        }
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )

    async def search_top_k(
        self,
        embedding: List[float],
        entity_type: str = "person",
        top_k: int = 5
    ) -> List[tuple[str, float]]:
        """
        Searches the embedding gallery for the top K similar vectors.
        Returns a list of (entity_id, score) tuples.
        """
        search_result = await self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="entity_type",
                        match=models.MatchValue(value=entity_type),
                    )
                ]
            ),
            limit=top_k
        )
        
        # Extract entity_id and similarity score
        results = []
        for hit in search_result:
            entity_id = hit.payload.get("entity_id")
            if entity_id:
                results.append((entity_id, hit.score))
                
        return results

vector_repository = VectorRepository()
