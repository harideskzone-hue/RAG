import pytest
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.infrastructure.db.postgres.models import Base, CameraModel, VideoSegmentModel, TrackModel, EvidenceModel

# Use in-memory SQLite for testing SQLAlchemy schemas without a real Postgres instance
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

import pytest_asyncio

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
        
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_postgres_schema_creation(db_session):
    """Test that all tables can be created and basic foreign keys work."""
    # 1. Create Camera
    cam = CameraModel(camera_id="CAM_TEST", location="Lobby")
    db_session.add(cam)
    await db_session.commit()
    
    # 2. Create Video Segment
    vid = VideoSegmentModel(video_id="VID_001", camera_id="CAM_TEST")
    db_session.add(vid)
    await db_session.commit()
    
    # 3. Create Track
    track_uuid = uuid.uuid4()
    track = TrackModel(id=track_uuid, track_id="P001", video_id="VID_001")
    db_session.add(track)
    await db_session.commit()
    
    # 4. Create Evidence
    ev_id = uuid.uuid4()
    ev = EvidenceModel(
        evidence_id=ev_id,
        video_id="VID_001",
        camera_id="CAM_TEST",
        track_uuid=track_uuid,
        source_type="video_ingestion",
        confidence=0.95
    )
    db_session.add(ev)
    await db_session.commit()
    
    # Retrieve and check relationships
    from sqlalchemy.future import select
    result = await db_session.execute(select(EvidenceModel).where(EvidenceModel.evidence_id == ev_id))
    fetched_ev = result.scalars().first()
    
    assert fetched_ev is not None
    assert fetched_ev.video_id == "VID_001"
    assert fetched_ev.confidence == 0.95
    
    print("PostgreSQL schema validation passed!")
