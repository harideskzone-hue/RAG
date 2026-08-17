import pytest
from unittest.mock import AsyncMock, patch

from app.infrastructure.db.mongodb.client import MongoDBClient

@pytest.mark.asyncio
async def test_mongo_index_creation():
    """Verify that the required temporal and identity indexes are created."""
    with patch("motor.motor_asyncio.AsyncIOMotorClient") as mock_motor_client:
        mock_db = AsyncMock()
        mock_col = AsyncMock()
        
        mock_motor_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_col
        
        client = MongoDBClient()
        # Override the mocked collection to make sure we test our mocked methods
        client.observations_col = mock_col
        
        await client.setup_indexes()
        
        # Check that create_index was called 4 times for the required indexes
        assert mock_col.create_index.call_count == 4
        
        calls = mock_col.create_index.call_args_list
        index_keys = [call[0][0] for call in calls]
        
        assert [("evidence_id", 1)] in index_keys
        assert [("video_id", 1)] in index_keys
        assert [("camera_id", 1), ("timestamp_sec", 1)] in index_keys
        assert [("track_id", 1), ("timestamp_sec", 1)] in index_keys
        
        print("MongoDB index validation passed!")
