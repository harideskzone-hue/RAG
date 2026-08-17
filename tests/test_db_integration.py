import pytest
import uuid
from unittest.mock import AsyncMock

from app.schemas.evidence_contract import EvidenceContract, EvidenceProvenance, EvidenceSubject
from app.services.db_services import EvidenceService

@pytest.mark.asyncio
async def test_evidence_service_idempotent_integration():
    """
    Test that EvidenceService coordinates the Postgres and Mongo boundaries correctly.
    """
    mock_pg_repo = AsyncMock()
    mock_mongo_repo = AsyncMock()
    
    service = EvidenceService(evidence_repo=mock_pg_repo, obs_repo=mock_mongo_repo)
    
    # Create an evidence contract
    ev = EvidenceContract(
        evidence_id=uuid.uuid4(),
        provenance=EvidenceProvenance(
            video_id="VID_001",
            camera_id="CAM_TEST",
            track_id="P001",
            source_type="video_ingestion",
        ),
        subject=EvidenceSubject(
            entity_type="person",
            track_id="P001",
        ),
        confidence=0.99
    )
    
    observation_data = {
        "evidence_id": str(ev.evidence_id),
        "video_id": "VID_001",
        "camera_id": "CAM_TEST",
        "track_id": "P001",
        "frame_index": 120,
        "timestamp_sec": 4.0,
        "bbox": [10, 20, 100, 200],
        "confidence": 0.99,
        "crop_uri": f"tracks/VID_001/P001/crops/{ev.evidence_id}.jpg"
    }
    
    await service.save_evidence(ev, observation_data)
    
    # Verify separation of concerns
    mock_pg_repo.create.assert_called_once_with(ev)
    mock_mongo_repo.insert_observation.assert_called_once_with(observation_data)
    
    print("Database integration transaction passed!")


@pytest.mark.asyncio
async def test_evidence_service_idempotency_retries():
    """Verify that calling save_evidence twice relies on repository idempotency."""
    mock_pg_repo = AsyncMock()
    mock_mongo_repo = AsyncMock()
    service = EvidenceService(evidence_repo=mock_pg_repo, obs_repo=mock_mongo_repo)
    
    ev = EvidenceContract()
    obs = {"evidence_id": str(ev.evidence_id)}
    
    # Call twice
    await service.save_evidence(ev, obs)
    await service.save_evidence(ev, obs)
    
    # Service should just call the repos twice, relying on their internal UPSERTs
    assert mock_pg_repo.create.call_count == 2
    assert mock_mongo_repo.insert_observation.call_count == 2


@pytest.mark.asyncio
async def test_evidence_service_cross_store_failure():
    """Verify that if Mongo fails, the error is propagated and NOT silently ignored."""
    mock_pg_repo = AsyncMock()
    mock_mongo_repo = AsyncMock()
    service = EvidenceService(evidence_repo=mock_pg_repo, obs_repo=mock_mongo_repo)
    
    ev = EvidenceContract()
    obs = {"evidence_id": str(ev.evidence_id)}
    
    # Simulate Mongo failure
    mock_mongo_repo.insert_observation.side_effect = Exception("MongoDB connection failed")
    
    with pytest.raises(Exception, match="MongoDB connection failed"):
        await service.save_evidence(ev, obs)
    
    # Postgres was called, Mongo failed, error bubbled up.
    mock_pg_repo.create.assert_called_once()
    mock_mongo_repo.insert_observation.assert_called_once()


@pytest.mark.asyncio
async def test_evidence_authority_and_temporal_retrieval():
    """Verify that temporal retrieval searches Postgres and ignores Mongo."""
    mock_pg_repo = AsyncMock()
    mock_mongo_repo = AsyncMock()
    service = EvidenceService(evidence_repo=mock_pg_repo, obs_repo=mock_mongo_repo)
    
    # Simulate Postgres returning canonical evidence
    ev = EvidenceContract()
    mock_pg_repo.search.return_value = [ev]
    
    # Perform temporal search
    results = await service.search_evidence(
        video_id="VID_001",
        start_time_sec=10.0,
        end_time_sec=20.0
    )
    
    assert len(results) == 1
    assert results[0] == ev
    
    # Postgres was called with temporal bounds
    mock_pg_repo.search.assert_called_once_with("VID_001", None, None, 10.0, 20.0)
    
    # Mongo was NOT called for evidence retrieval (Authority check passed)
    mock_mongo_repo.get_observations_for_evidence.assert_not_called()
