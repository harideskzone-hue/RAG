import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.schemas.event_contract import VerifiedEventContract
from app.tools.vector.store import get_vector_store

logger = logging.getLogger("EventRepository")


class EventRepository:
    """
    Manages persistent multi-store storage and deterministic querying for surveillance events
    across PostgreSQL, MongoDB, Qdrant (event_embeddings_v1), and Object Storage.
    """

    def __init__(self):
        self.vector_store = get_vector_store()

    async def init_postgres_schema(self, session):
        """
        Ensures PostgreSQL events table exists.
        """
        query = """
        CREATE TABLE IF NOT EXISTS events (
            event_id VARCHAR(64) PRIMARY KEY,
            event_type VARCHAR(64) NOT NULL,
            camera_id VARCHAR(64) NOT NULL,
            video_id VARCHAR(256) NOT NULL,
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            duration_sec FLOAT NOT NULL,
            track_ids JSONB DEFAULT '[]'::jsonb,
            canonical_person_ids JSONB DEFAULT '[]'::jsonb,
            confidence FLOAT NOT NULL,
            severity VARCHAR(32) NOT NULL,
            clip_path VARCHAR(512),
            clip_url VARCHAR(512),
            thumbnail_path VARCHAR(512),
            thumbnail_url VARCHAR(512),
            reason TEXT,
            clip_sha256 VARCHAR(64),
            provenance JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
        CREATE INDEX IF NOT EXISTS idx_events_video ON events(video_id);
        """
        from sqlalchemy import text
        await session.execute(text(query))
        await session.commit()

    async def save_event(self, event: VerifiedEventContract, db_session=None, mongo_db=None):
        """
        Persists a verified event into PostgreSQL, MongoDB, and Qdrant (event_embeddings_v1).
        """
        # 1. PostgreSQL Persistence
        if db_session:
            try:
                await self.init_postgres_schema(db_session)
                from sqlalchemy import text
                insert_query = text("""
                INSERT INTO events (
                    event_id, event_type, camera_id, video_id, start_time, end_time,
                    duration_sec, track_ids, canonical_person_ids, confidence, severity,
                    clip_path, clip_url, thumbnail_path, thumbnail_url, reason, clip_sha256, provenance
                ) VALUES (
                    :event_id, :event_type, :camera_id, :video_id, :start_time, :end_time,
                    :duration_sec, :track_ids, :canonical_person_ids, :confidence, :severity,
                    :clip_path, :clip_url, :thumbnail_path, :thumbnail_url, :reason, :clip_sha256, :provenance
                ) ON CONFLICT (event_id) DO UPDATE SET
                    event_type = EXCLUDED.event_type,
                    confidence = EXCLUDED.confidence,
                    clip_sha256 = EXCLUDED.clip_sha256;
                """)
                await db_session.execute(insert_query, {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "camera_id": event.camera_id,
                    "video_id": event.video_id,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "duration_sec": event.duration_sec,
                    "track_ids": json.dumps(event.track_ids),
                    "canonical_person_ids": json.dumps(event.canonical_person_ids),
                    "confidence": event.confidence,
                    "severity": event.severity,
                    "clip_path": event.clip_path,
                    "clip_url": event.clip_url,
                    "thumbnail_path": event.thumbnail_path,
                    "thumbnail_url": event.thumbnail_url,
                    "reason": event.reason,
                    "clip_sha256": event.clip_sha256,
                    "provenance": json.dumps(event.provenance)
                })
                await db_session.commit()
            except Exception as e:
                logger.error(f"Failed to persist event to PostgreSQL: {e}")

        # 2. MongoDB Persistence
        if mongo_db is not None:
            try:
                event_dict = event.dict()
                event_dict["created_at"] = datetime.utcnow()
                await mongo_db["events"].replace_one(
                    {"event_id": event.event_id},
                    event_dict,
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Failed to persist event to MongoDB: {e}")

        # 3. Qdrant event_embeddings_v1 Persistence
        try:
            from app.tools.vector.encoder import get_vector_encoder
            encoder = get_vector_encoder()
            event_text = f"Incident: {event.event_type}. Reason: {event.reason}. Camera: {event.camera_id}. Severity: {event.severity}."
            event_vec = encoder.encode(event_text)

            v_data = [
                [event.event_id],
                [event_vec],
                [event.camera_id],
                [str(event.start_time)],
                [event_text]
            ]
            await self.vector_store.insert("event_embeddings_v1", v_data)
        except Exception as e:
            logger.warning(f"Could not index event in Qdrant event_embeddings_v1: {e}")

    async def get_events(
        self,
        event_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        video_id: Optional[str] = None,
        canonical_person_id: Optional[str] = None,
        limit: int = 10,
        db_session = None
    ) -> List[VerifiedEventContract]:
        """
        Deterministic event retrieval tool query.
        """
        events = []
        if db_session:
            try:
                from sqlalchemy import text
                where_clauses = ["1=1"]
                params = {"limit": limit}

                if event_type and event_type.upper() != "ALL":
                    where_clauses.append("UPPER(event_type) = :event_type")
                    params["event_type"] = event_type.upper()
                if camera_id:
                    where_clauses.append("camera_id = :camera_id")
                    params["camera_id"] = camera_id
                if video_id:
                    where_clauses.append("video_id = :video_id")
                    params["video_id"] = video_id

                query_str = f"SELECT * FROM events WHERE {' AND '.join(where_clauses)} ORDER BY start_time ASC LIMIT :limit"
                result = await db_session.execute(text(query_str), params)
                rows = result.mappings().all()

                for row in rows:
                    pids = row["canonical_person_ids"]
                    if isinstance(pids, str):
                        pids = json.loads(pids)
                    tids = row["track_ids"]
                    if isinstance(tids, str):
                        tids = json.loads(tids)
                    prov = row["provenance"]
                    if isinstance(prov, str):
                        prov = json.loads(prov)

                    if canonical_person_id and canonical_person_id not in pids:
                        continue

                    events.append(VerifiedEventContract(
                        event_id=row["event_id"],
                        event_type=row["event_type"],
                        camera_id=row["camera_id"],
                        video_id=row["video_id"],
                        start_time=row["start_time"],
                        end_time=row["end_time"],
                        duration_sec=row["duration_sec"],
                        track_ids=tids,
                        canonical_person_ids=pids,
                        confidence=row["confidence"],
                        severity=row["severity"],
                        clip_path=row["clip_path"] or "",
                        clip_url=row["clip_url"] or f"/media/events/{row['event_id']}/clip.mp4",
                        thumbnail_path=row["thumbnail_path"] or "",
                        thumbnail_url=row["thumbnail_url"] or f"/media/events/{row['event_id']}/thumbnail.jpg",
                        reason=row["reason"] or "",
                        clip_sha256=row["clip_sha256"] or "",
                        provenance=prov or {}
                    ))
            except Exception as e:
                logger.error(f"Error querying events from PostgreSQL: {e}")

        # Fallback to local files in dataset/events/ if DB offline or empty
        if not events:
            from pathlib import Path
            events_root = Path("dataset/events")
            if events_root.exists():
                for edir in sorted(events_root.iterdir()):
                    if edir.is_dir() and (edir / "event.json").exists():
                        try:
                            with open(edir / "event.json") as f:
                                meta = json.load(f)
                            e_type = meta.get("metadata", {}).get("event_type", "LOITERING")
                            if event_type and event_type.upper() not in ["ALL", e_type.upper()]:
                                continue
                            events.append(VerifiedEventContract(
                                event_id=edir.name,
                                event_type=e_type,
                                camera_id=meta.get("camera_id", "cam_auto_01"),
                                video_id=meta.get("source_video_id", ""),
                                start_time=meta.get("target_event_start", 0.0),
                                end_time=meta.get("target_event_end", 0.0),
                                duration_sec=round(meta.get("target_event_end", 0.0) - meta.get("target_event_start", 0.0), 2),
                                track_ids=meta.get("metadata", {}).get("track_ids", []),
                                canonical_person_ids=meta.get("metadata", {}).get("canonical_person_ids", []),
                                confidence=meta.get("metadata", {}).get("confidence", 0.9),
                                severity=meta.get("metadata", {}).get("severity", "MEDIUM"),
                                clip_path=str(edir / "clip.mp4"),
                                clip_url=f"/media/events/{edir.name}/clip.mp4",
                                thumbnail_path=str(edir / "thumbnail.jpg"),
                                thumbnail_url=f"/media/events/{edir.name}/thumbnail.jpg",
                                reason=meta.get("metadata", {}).get("reason", "Incident event verified from storage"),
                                clip_sha256=meta.get("clip_sha256", ""),
                                provenance=meta
                            ))
                        except Exception:
                            pass

        return events[:limit]
