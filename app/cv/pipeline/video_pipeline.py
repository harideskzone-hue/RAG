import logging
from typing import List, Optional
from pathlib import Path

from app.cv.models.registry import ModelRegistry
from app.cv.ingestion.video_reader import VideoReader
from app.cv.sampling.sampler import FrameSampler
from app.cv.detection.yolo import YOLOPersonDetector
from app.cv.tracks.aggregator import TrackAggregator
from app.cv.evidence.builder import EvidenceBuilder
from app.cv.interfaces.evidence_output import EvidenceOutputInterface
from app.schemas.evidence_contract import EvidenceContract

logger = logging.getLogger(__name__)

class VideoPipeline:
    """End-to-End Computer Vision Pipeline for Phase 1."""

    def __init__(
        self, 
        config_overrides: dict = None,
        output_interface: Optional[EvidenceOutputInterface] = None,
        crop_dir: str = "dataset/tracks"
    ):
        self.registry = ModelRegistry(config_overrides)
        self.registry.validate()
        
        self.sampler = FrameSampler()
        self.detector = YOLOPersonDetector(self.registry)
        self.crop_dir = crop_dir
        self.output_interface = output_interface

    def process_video(self, video_path: str, video_id: str, camera_id: str, progress_callback=None) -> List[EvidenceContract]:
        """Processes a single video and yields EvidenceContracts."""
        logger.info(f"Starting CV pipeline on {video_path} (ID: {video_id}, Camera: {camera_id})")
        
        reader = VideoReader(video_path, video_id)
        video_id = reader.video_id
        
        # Reset detector tracker state for the new video
        self.detector.reset()
        
        aggregator = TrackAggregator(video_id=video_id, camera_id=camera_id, output_dir=self.crop_dir)
        
        # Sample frames
        frames = self.sampler.sample(reader.native_fps, reader.read_frames())
        
        total_frames = reader.total_frames
        
        for i, (frame_index, frame_bgr, timestamp_sec) in enumerate(frames):
            if progress_callback:
                progress_callback(frame_index, total_frames)
                
            detections = self.detector.track_frame(frame_bgr)
            if detections:
                aggregator.process_frame(
                    frame_bgr=frame_bgr,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    detections=detections
                )
                
        if progress_callback:
            progress_callback(total_frames, total_frames)
                
        # Build evidence
        all_observations = aggregator.get_all_evidence()
        contracts = EvidenceBuilder.build_from_observations(all_observations)
        
        logger.info(f"Video {video_id} processing complete. Extracted {len(contracts)} observations.")
        
        if self.output_interface:
            self.output_interface.push_evidence(contracts)
            
        return contracts
