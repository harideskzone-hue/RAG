import pytest
from unittest.mock import AsyncMock, patch
from qdrant_client.http import models

from app.infrastructure.db.qdrant.client import VectorRepository

@pytest.mark.asyncio
async def test_qdrant_synthetic_vector_insertion():
    """Verify that vector insertion works using ONLY synthetic vectors (0.1, 0.2, etc)."""
    with patch("app.infrastructure.db.qdrant.client.AsyncQdrantClient") as mock_qdrant_client_class:
        mock_client = AsyncMock()
        mock_qdrant_client_class.return_value = mock_client
        
        repo = VectorRepository()
        repo.client = mock_client
        
        # Ensure collection exists logic works
        mock_client.collection_exists.return_value = False
        await repo.setup_collection()
        mock_client.create_collection.assert_called_once()
        assert mock_client.create_payload_index.call_count == 3
        
        # Test synthetic vector insertion
        synthetic_vector = [0.1] * 512
        await repo.insert_embedding(
            vector_id="TEST-VEC-1",
            embedding=synthetic_vector,
            entity_type="person",
            entity_id="P001",
            video_id="VID_001",
            camera_id="CAM_TEST",
            track_id="P001",
        )
        
        mock_client.upsert.assert_called_once()
        args, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "vista_embeddings"
        
        points = kwargs["points"]
        assert len(points) == 1
        assert points[0].id == "TEST-VEC-1"
        assert points[0].vector == synthetic_vector
        assert points[0].payload["entity_type"] == "person"
        assert points[0].payload["entity_id"] == "P001"
        
        print("Qdrant synthetic vector insertion passed!")
