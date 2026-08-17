import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from app.cv.identity.quality import CropQualitySelector
from app.cv.reid.osnet import OSNetExtractor
from app.cv.identity.resolver import IdentityResolver, ResolutionStatus

@pytest.mark.asyncio
async def test_full_cv_to_db_identity_flow():
    """
    Validates the end-to-end flow: Track Crop -> Quality Check -> OSNet -> Qdrant -> Resolver -> Postgres/Blob
    """
    # 1. Quality Check
    selector = CropQualitySelector()
    np.random.seed(123)
    good_crop = np.random.randint(50, 200, (256, 128, 3), dtype=np.uint8)
    quality_res = selector.assess_quality(good_crop)
    assert quality_res["approved"] is True
    
    # 2. OSNet Embedding
    extractor = OSNetExtractor()
    embedding = extractor.extract(good_crop)
    
    # 3. Qdrant Similarity Search
    mock_qdrant = AsyncMock()
    mock_qdrant.search_top_k.return_value = [("P-OLD123", 0.95)]  # Mock high similarity
    
    search_results = await mock_qdrant.search_top_k(embedding)
    
    # 4. Identity Resolver
    resolver = IdentityResolver()
    status, person_id = resolver.resolve(search_results)
    
    assert status == ResolutionStatus.MATCHED
    assert person_id == "P-OLD123"
    
    # 5. Database & Storage updates
    mock_pg_person_repo = AsyncMock()
    mock_pg_track_repo = AsyncMock()
    mock_blob = AsyncMock()
    
    # Assign the track to the resolved person
    await mock_pg_track_repo.assign_person_to_track(
        video_id="VID_001", 
        track_id="T001", 
        person_id=person_id
    )
    
    # Copy the crop
    await mock_blob.copy_crop_to_person(
        original_uri="tracks/VID_001/T001/crops/crop_1.jpg",
        person_id=person_id,
        crop_id="crop_1"
    )
    
    mock_pg_track_repo.assign_person_to_track.assert_called_once_with(
        video_id="VID_001", 
        track_id="T001", 
        person_id="P-OLD123"
    )
    mock_blob.copy_crop_to_person.assert_called_once()
