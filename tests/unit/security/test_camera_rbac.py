import pytest
from unittest.mock import AsyncMock
import numpy as np
from app.schemas.context import VistaContext
from app.services.repositories.camera_repository import CameraRepository
from app.services.repositories.alert_repository import AlertRepository
from app.tools.vector.store import NativeVectorStore

@pytest.mark.asyncio
async def test_unauthorized_camera_returns_zero_evidence():
    # 1. Test Metadata RBAC (Camera)
    mock_db = AsyncMock()
    mock_db.execute.return_value.success = True
    mock_db.execute.return_value.rows = [{"id": "camera_02", "location": "Entrance", "status": "active"}] # DB has camera_02
    
    repo = CameraRepository(mock_db)
    
    # User only allowed camera_01
    from app.schemas.context import UserContext
    context = VistaContext(
        user=UserContext(user_id="test_user", role="investigator", allowed_cameras=["camera_01"]), 
        conversation_id="test_conv", 
        current_query="find camera_02",
    )
    
    # Trying to get camera_02 should return None due to RBAC
    result = await repo.get_camera("camera_02", context)
    assert result is None
    
    # Getting all cameras should only fetch camera_01 (check the query)
    await repo.get_all_cameras(context)
    args, kwargs = mock_db.execute.call_args
    assert "WHERE id IN ($1)" in kwargs["query"]
    assert kwargs["params"] == ["camera_01"]
    
    # 2. Test Vector Store RBAC
    store = NativeVectorStore(data_dir="/tmp/test_rbac")
    store.vectors = np.array([[0.1, 0.2]])
    store.metadata = [{"id": "1", "camera_id": "camera_02", "timestamp": "2023", "description": "test", "bbox": None}]
    
    # Search for vector, but user only has access to camera_01
    matches = store._search("test_col", [0.1, 0.2], top_k=5, allowed_cameras=["camera_01"])
    assert len(matches) == 0 # Should return 0 matches
    
    # If user has access to camera_02, it should return matches
    matches = store._search("test_col", [0.1, 0.2], top_k=5, allowed_cameras=["camera_02"])
    assert len(matches) == 1
