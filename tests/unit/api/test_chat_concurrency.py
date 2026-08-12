import asyncio
import pytest
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app.app import app
from app.api.dependencies.security import get_current_user
from fastapi import Request

client = TestClient(app)

async def override_get_current_user(request: Request):
    cam = request.headers.get("X-Test-Camera", "CAM_02")
    return {"sub": "1", "role": "admin", "allowed_cameras": [cam]}

@pytest.mark.asyncio
async def test_chat_concurrency():
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    import httpx
    
    # Mock the reasoning LLM to prevent loading Qwen concurrently in tests
    from unittest.mock import patch
    from app.domain.models.reasoning import ReasoningResult, Hypothesis
    from app.agents.reasoning.engine.reasoning_coordinator import ReasoningCoordinator
    from app.tools.vector.encoder import ModelFreeVectorEncoder
    
    # Just mock execute to return a generic success
    async def mock_execute(self, context):
        return ReasoningResult(success=True, answer="Mocked answer", claims=[], uncertainties=[])
        
    with patch.object(ReasoningCoordinator, 'execute', new=mock_execute):
        with patch('app.tools.vector.encoder.get_vector_encoder', return_value=ModelFreeVectorEncoder()):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
                tasks = []
                for i in range(10):
                    cam_id = f"CAM_0{2 if i % 2 == 0 else 3}"
                    
                    tasks.append(
                        ac.post(
                            "/api/v1/chat",
                            json={"query": f"find person {i}"},
                            headers={"Authorization": "Bearer dummy", "X-Test-Camera": cam_id}
                        )
                    )
                responses = await asyncio.gather(*tasks)
                
                for i, response in enumerate(responses):
                    assert response.status_code == 200
                    data = response.json()
                    cam_id = f"CAM_0{2 if i % 2 == 0 else 3}"
                    
                    if cam_id == "CAM_03":
                        assert data["status"] == "NO_AUTHORIZED_EVIDENCE"
                        assert len(data["evidence"]) == 0
                    else:
                        for ev in data["evidence"]:
                            assert ev["camera_id"] == "CAM_02"
                        
                trace_ids = [r.json()["trace_id"] for r in responses]
                assert len(set(trace_ids)) == 10, "Trace IDs were not unique across concurrent requests!"
            
    app.dependency_overrides.clear()
