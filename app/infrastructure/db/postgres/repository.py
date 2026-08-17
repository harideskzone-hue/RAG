from typing import Any, List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.dialects.postgresql import insert

from app.domain.repositories.base import EvidenceRepository, PersonRepository, TrackRepository
from app.schemas.evidence_contract import EvidenceContract
from app.infrastructure.db.postgres.models import EvidenceModel, TrackModel, VideoSegmentModel, CameraModel

class PostgresEvidenceRepository(EvidenceRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create(self, evidence: EvidenceContract) -> None:
        async with self.session_factory() as session:
            # 1. Upsert Camera if provided
            cam_id = evidence.provenance.camera_id or "cam_auto_01"
            cam_stmt = insert(CameraModel).values(
                camera_id=cam_id,
                location="Main",
                resolution="1920x1080"
            ).on_conflict_do_nothing(index_elements=['camera_id'])
            await session.execute(cam_stmt)

            # 2. Upsert VideoSegment
            vid_id = evidence.provenance.video_id or "default_video_id"
            vid_stmt = insert(VideoSegmentModel).values(
                video_id=vid_id,
                camera_id=cam_id,
                duration_sec=111.0
            ).on_conflict_do_nothing(index_elements=['video_id'])
            await session.execute(vid_stmt)

            # 3. Upsert track metadata
            track_stmt = insert(TrackModel).values(
                track_id=evidence.subject.track_id,
                video_id=vid_id,
            ).on_conflict_do_nothing(
                index_elements=['video_id', 'track_id']
            ).returning(TrackModel.id)
            
            result = await session.execute(track_stmt)
            track_uuid = result.scalar()
            
            if not track_uuid:
                # Track already exists, get its UUID
                sel_stmt = select(TrackModel.id).where(
                    TrackModel.video_id == vid_id,
                    TrackModel.track_id == evidence.subject.track_id
                )
                track_uuid = (await session.execute(sel_stmt)).scalar()

            # Upsert Evidence (Idempotency: DO NOTHING if evidence_id exists)
            ev_stmt = insert(EvidenceModel).values(
                evidence_id=evidence.evidence_id,
                video_id=evidence.provenance.video_id,
                camera_id=evidence.provenance.camera_id,
                track_uuid=track_uuid,
                source_type=evidence.provenance.source_type,
                confidence=evidence.confidence,
                attributes=evidence.attributes.model_dump(),
                description=evidence.subject.description
            ).on_conflict_do_nothing(
                index_elements=['evidence_id']
            )
            
            await session.execute(ev_stmt)
            await session.commit()

    async def get_by_id(self, evidence_id: str | UUID) -> Optional[EvidenceContract]:
        pass  # Omitted for brevity

    async def search(
        self, 
        video_id: str, 
        camera_id: Optional[str] = None, 
        track_id: Optional[str] = None,
        start_time_sec: Optional[float] = None,
        end_time_sec: Optional[float] = None
    ) -> List[EvidenceContract]:
        """
        Authoritative temporal retrieval using PostgreSQL.
        Mongo observations are NOT used here to preserve contract authority.
        """
        async with self.session_factory() as session:
            stmt = select(EvidenceModel).join(TrackModel).where(EvidenceModel.video_id == video_id)
            
            if camera_id:
                stmt = stmt.where(EvidenceModel.camera_id == camera_id)
            if track_id:
                stmt = stmt.where(TrackModel.track_id == track_id)
            if start_time_sec is not None:
                stmt = stmt.where(TrackModel.first_seen_sec >= start_time_sec)
            if end_time_sec is not None:
                stmt = stmt.where(TrackModel.last_seen_sec <= end_time_sec)
                
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            # Map back to EvidenceContract (mock implementation for test)
            return [EvidenceContract(evidence_id=rec.evidence_id) for rec in records]


class PostgresPersonRepository(PersonRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory
        
    async def create_person(self) -> str:
        async with self.session_factory() as session:
            from app.infrastructure.db.postgres.models import CanonicalPersonModel
            person = CanonicalPersonModel()
            session.add(person)
            await session.commit()
            return person.person_id


class PostgresTrackRepository(TrackRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory
        
    async def get_track_summary(self, video_id: str, track_id: str) -> Optional[dict[str, Any]]:
        pass
        
    async def assign_person_to_track(self, video_id: str, track_id: str, person_id: str) -> None:
        async with self.session_factory() as session:
            from sqlalchemy import update
            from app.infrastructure.db.postgres.models import TrackModel
            stmt = update(TrackModel).where(
                TrackModel.video_id == video_id,
                TrackModel.track_id == track_id
            ).values(person_id=person_id)
            await session.execute(stmt)
            await session.commit()
