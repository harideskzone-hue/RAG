import uuid
from typing import Any
from sqlalchemy import Column, String, Float, Integer, Boolean, ForeignKey, DateTime, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class CameraModel(Base):
    __tablename__ = "cameras"

    camera_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    rtsp_url = Column(String, nullable=True)
    location = Column(String, nullable=True)
    resolution = Column(String, nullable=True)
    enabled = Column(Boolean, default=True)
    expected_fps = Column(Float, default=30.0)
    segment_duration = Column(Integer, default=600)  # 10 minutes in seconds
    status = Column(String, default="ACTIVE")
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    videos = relationship("VideoSegmentModel", back_populates="camera")


class VideoSegmentModel(Base):
    __tablename__ = "video_segments"

    video_id = Column(String, primary_key=True)  # segment_id
    camera_id = Column(String, ForeignKey("cameras.camera_id"), nullable=False, index=True)
    file_name = Column(String, nullable=True)
    sha256 = Column(String, nullable=True, index=True)
    status = Column(String, default="READY")  # RECORDING, READY, PROCESSING, CV_COMPLETE, PERSISTING, VERIFYING, VERIFIED, CLEANUP_PENDING, COMPLETED, FAILED
    cleanup_status = Column(String, default="PENDING")  # PENDING, DELETED, RETAINED
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_sec = Column(Float, nullable=True)
    expected_frames = Column(Integer, default=0)
    received_frames = Column(Integer, default=0)
    dropped_frames = Column(Integer, default=0)
    fps = Column(Float, nullable=True)
    storage_uri = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)

    camera = relationship("CameraModel", back_populates="videos")
    tracks = relationship("TrackModel", back_populates="video")


class CanonicalPersonModel(Base):
    __tablename__ = "canonical_persons"

    person_id = Column(String, primary_key=True, default=lambda: f"P-{uuid.uuid4().hex[:8].upper()}")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    tracks = relationship("TrackModel", back_populates="person")


class TrackModel(Base):
    __tablename__ = "tracks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(String, nullable=False)  # CV local track ID (e.g., P001)
    video_id = Column(String, ForeignKey("video_segments.video_id"), nullable=False)
    person_id = Column(String, ForeignKey("canonical_persons.person_id"), nullable=True)
    
    first_seen_sec = Column(Float, nullable=True)
    last_seen_sec = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    video = relationship("VideoSegmentModel", back_populates="tracks")
    person = relationship("CanonicalPersonModel", back_populates="tracks")
    evidence = relationship("EvidenceModel", back_populates="track_meta")

    __table_args__ = (
        UniqueConstraint('video_id', 'track_id', name='uix_video_track'),
    )


class EvidenceModel(Base):
    __tablename__ = "evidence"

    evidence_id = Column(UUID(as_uuid=True), primary_key=True)
    
    video_id = Column(String, ForeignKey("video_segments.video_id"), nullable=False)
    camera_id = Column(String, ForeignKey("cameras.camera_id"), nullable=False)
    
    track_uuid = Column(UUID(as_uuid=True), ForeignKey("tracks.id"), nullable=False)
    
    source_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    
    attributes = Column(JSON, nullable=True)
    description = Column(String, nullable=True)
    
    crop_uri = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    track_meta = relationship("TrackModel", back_populates="evidence")
