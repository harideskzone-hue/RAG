import logging
import uuid
import cv2
from typing import List, Dict, Any

from app.cv.pipeline.video_pipeline import VideoPipeline
from app.cv.identity.quality import CropQualitySelector
from app.cv.identity.resolver import IdentityResolver, ResolutionStatus
from app.services.repositories.person_repository import PersonRepository
from app.domain.repositories.base import ObservationRepository
from app.schemas.evidence_contract import EvidenceContract, EvidenceSubject, EvidenceProvenance, EvidenceAttributes
from app.schemas.context import VistaContext
from app.platform.config.config import config

logger = logging.getLogger(__name__)

class IngestionManager:
    """
    Orchestrates the CV pipeline for background video ingestion:
    YOLO26n -> ByteTrack -> Crop Quality -> OSNet -> IdentityResolver -> Persistence.
    """
    def __init__(self, person_repo: PersonRepository, obs_repo: ObservationRepository, evidence_repo=None):
        self.person_repo = person_repo
        self.obs_repo = obs_repo
        
        if evidence_repo is None:
            try:
                from app.infrastructure.db.postgres.repository import PostgresEvidenceRepository
                from app.config.db import db_settings
                from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
                from sqlalchemy.orm import sessionmaker
                engine = create_async_engine(db_settings.POSTGRES_URI)
                session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
                self.evidence_repo = PostgresEvidenceRepository(session_factory)
            except Exception as e:
                logger.warning(f"Could not initialize PostgresEvidenceRepository: {e}")
                self.evidence_repo = None
        else:
            self.evidence_repo = evidence_repo

        try:
            from app.cv.reid.osnet import OSNetExtractor
            self.reid_extractor = OSNetExtractor()
        except Exception as e:
            logger.warning(f"Could not load OSNetExtractor: {e}")
            self.reid_extractor = None
        
        self.quality_selector = CropQualitySelector()
        self.identity_resolver = IdentityResolver()
        self.pipeline = VideoPipeline()
        
    async def process_and_persist(self, video_path: str, video_id: str, camera_id: str, progress_callback=None, contracts: List[EvidenceContract] = None):
        """
        Runs the video through the full CV pipeline, generates embeddings, 
        resolves identities against the vector store, and persists them.
        """
        logger.info(f"IngestionManager: Starting persistence for {video_id}")
        
        import asyncio
        # 1. YOLO26n + ByteTrack + Cropping (Existing Pipeline if not already extracted)
        if contracts is None:
            contracts = await asyncio.to_thread(self.pipeline.process_video, video_path, video_id, camera_id, progress_callback)
        
        logger.info(f"IngestionManager: Extracted/provided {len(contracts)} track observations.")
        
        # Group observations by track_id
        track_obs = {}
        for c in contracts:
            t_id = c.subject.track_id
            if t_id not in track_obs:
                track_obs[t_id] = []
            track_obs[t_id].append(c)
            
        from app.schemas.context import UserContext
        user = UserContext(user_id="system", role="system")
        context = VistaContext(conversation_id=f"ingest_{video_id}", user=user)
        resolved_pids = {}
        
        # 2. Quality Selection, OSNet Re-ID, and Identity Resolution per track
        total_tracks = len(track_obs)
        for t_idx, (track_id, observations) in enumerate(track_obs.items(), 1):
            if progress_callback:
                try:
                    progress_callback(t_idx, total_tracks, stage="reid", detail=f"Track {track_id} ({t_idx}/{total_tracks})")
                except TypeError:
                    progress_callback(t_idx, total_tracks)
            best_crop = None
            best_quality = -1.0
            best_obs = None
            
            fallback_crop = None
            fallback_obs = None
            fallback_score = -1.0

            # Evaluate representative crops per track with strict quality gating & deduplication
            approved_crops = []
            for obs in observations:
                import os
                evidence_id = obs.observation.get("original_evidence_id")
                if evidence_id and track_id:
                    s_vid = str(video_id or "")
                    s_tid = str(track_id)
                    s_eid = str(evidence_id)
                    candidate_paths = [
                        os.path.join("dataset", "tracks", s_vid, s_tid, "crops", f"{s_eid}.jpg"),
                        os.path.join("dataset", "tracks", s_vid, f"P{s_tid.replace('P', '').zfill(3)}", "crops", f"{s_eid}.jpg"),
                        os.path.join("dataset", "tracks", s_tid, "crops", f"{s_eid}.jpg"),
                        os.path.join("dataset", "persons", s_tid, "crops", f"{s_eid}.jpg")
                    ]
                    for crop_path in candidate_paths:
                        if os.path.exists(crop_path):
                            crop_bgr = cv2.imread(crop_path)
                            if crop_bgr is not None:
                                quality = self.quality_selector.assess_quality(crop_bgr)
                                score = float(quality.get("score", 0))
                                if quality.get("approved"):
                                    # Deduplication: avoid adding duplicate viewpoints for the same track
                                    is_dup = any(
                                        self.quality_selector.are_duplicate_crops(crop_bgr, prev[1])
                                        for prev in approved_crops
                                    )
                                    if not is_dup:
                                        approved_crops.append((score, crop_bgr, obs, evidence_id))
                                        if score > best_quality:
                                            best_quality = score
                                            best_crop = crop_bgr
                                            best_obs = obs
                                if (crop_bgr.shape[0] * crop_bgr.shape[1]) > fallback_score:
                                    fallback_score = float(crop_bgr.shape[0] * crop_bgr.shape[1])
                                    fallback_crop = crop_bgr
                                    fallback_obs = obs
                                    fallback_ev_id = evidence_id
                            break
                                
            # If no crop strictly passed the quality threshold, use fallback only for embedding if non-empty, but DO NOT save to person gallery
            if best_crop is None and fallback_crop is not None:
                best_crop = fallback_crop
                best_obs = fallback_obs
                                
            if best_crop is not None and best_obs is not None:
                # 3. OSNet Extractor
                if self.reid_extractor:
                    embedding = self.reid_extractor.extract(best_crop)
                else:
                    embedding = [0.1] * 512 # Mock embedding
                    
                # 4. Identity Resolver
                search_results = []
                try:
                    matches = await self.person_repo.search_person(embedding, top_k=5, context=context)
                    search_results = [(m.id, m.score) for m in matches]
                except Exception as s_err:
                    logger.warning(f"Vector similarity search skipped for track {track_id}: {s_err}")
                
                status, canonical_id = self.identity_resolver.resolve(search_results)
                
                from app.tools.vector.store import get_vector_store
                vstore = get_vector_store()

                if status == ResolutionStatus.MATCHED:
                    logger.info(f"Matched tracklet {track_id} to existing identity {canonical_id}")
                    final_id = canonical_id
                    # Incremental Gallery Update: Add this approved viewpoint to canonical person gallery
                    v_data = [
                        [final_id], [embedding], [camera_id], [best_obs.provenance.video_timestamp_sec]
                    ]
                    try:
                        await vstore.insert("person_embeddings_v2", v_data)
                    except Exception as v_err:
                        logger.warning(f"Vector gallery insert skipped for track {track_id}: {v_err}")
                elif status == ResolutionStatus.NEW:
                    final_id = f"PERSON_{uuid.uuid4().hex[:8].upper()}"
                    logger.info(f"Created new identity {final_id} for tracklet {track_id}")
                    # Initialize canonical person gallery with first embedding
                    v_data = [
                        [final_id], [embedding], [camera_id], [best_obs.provenance.video_timestamp_sec]
                    ]
                    try:
                        await vstore.insert("person_embeddings_v2", v_data)
                    except Exception as v_err:
                        logger.warning(f"Vector gallery insert skipped for track {track_id}: {v_err}")
                else:
                    logger.warning(f"Ambiguous identity for tracklet {track_id}, preserving UNRESOLVED.")
                    final_id = f"UNRESOLVED_{track_id}"
                
                # Copy ONLY high-quality, deduplicated, enhanced crops into canonical person gallery if resolved
                if not final_id.startswith("UNRESOLVED") and approved_crops:
                    import json
                    from pathlib import Path
                    from app.cv.crops.enhancer import CropEnhancer
                    
                    enhancer = CropEnhancer()
                    canonical_dir = Path("dataset") / "persons" / final_id
                    canonical_crops_dir = canonical_dir / "crops"
                    canonical_crops_dir.mkdir(parents=True, exist_ok=True)

                    # Read existing gallery images to prevent cross-track duplicates
                    existing_crops = []
                    for existing_path in canonical_crops_dir.glob("*.jpg"):
                        ex_bgr = cv2.imread(str(existing_path))
                        if ex_bgr is not None:
                            existing_crops.append(ex_bgr)
                    
                    # Sort approved crops by quality score descending, keep top 4-6 distinct crops
                    approved_crops.sort(key=lambda x: x[0], reverse=True)
                    
                    saved_ev_ids = []
                    for _, c_img, _, c_ev_id in approved_crops:
                        if len(saved_ev_ids) + len(existing_crops) >= 5:
                            break
                        if c_img is not None and c_ev_id:
                            is_dup = any(
                                self.quality_selector.are_duplicate_crops(c_img, ex)
                                for ex in existing_crops
                            )
                            if not is_dup:
                                enhanced_img = enhancer.enhance(c_img)
                                canonical_crop_path = canonical_crops_dir / f"{c_ev_id}.jpg"
                                cv2.imwrite(str(canonical_crop_path), enhanced_img)
                                saved_ev_ids.append(str(c_ev_id))
                                existing_crops.append(c_img)

                    # Maintain person metadata
                    person_meta_file = canonical_dir / "person.json"
                    person_data = {
                        "canonical_person_id": final_id,
                        "tracks": [],
                        "evidence_ids": [],
                        "cameras": []
                    }
                    if person_meta_file.exists():
                        try:
                            with open(person_meta_file, "r") as pf:
                                person_data = json.load(pf)
                        except Exception:
                            pass
                    if track_id not in person_data["tracks"]:
                        person_data["tracks"].append(track_id)
                    for eid in saved_ev_ids:
                        if eid not in person_data["evidence_ids"]:
                            person_data["evidence_ids"].append(eid)
                    if camera_id not in person_data["cameras"]:
                        person_data["cameras"].append(camera_id)
                    
                    with open(person_meta_file, "w") as pf:
                        json.dump(person_data, pf, indent=2)

                # Set canonical person ID on contract
                best_obs.subject.track_id = track_id
                resolved_pids[track_id] = final_id
                
                # 5. MongoDB Observation Persistence
                if self.obs_repo and getattr(self, "_mongo_available", True):
                    try:
                        obs_data = {
                            "evidence_id": str(best_obs.evidence_id),
                            "video_id": video_id,
                            "camera_id": camera_id,
                            "canonical_person_id": final_id,
                            "original_track_id": track_id,
                            "timestamp": best_obs.provenance.video_timestamp_sec
                        }
                        await self.obs_repo.insert_observation(obs_data)
                    except Exception as obs_err:
                        self._mongo_available = False
                        logger.warning(f"MongoDB offline/unreachable — skipping DB observation persistence for subsequent tracks: {obs_err}")

                # 6. PostgreSQL Evidence Persistence
                if self.evidence_repo and getattr(self, "_pg_available", True):
                    try:
                        await self.evidence_repo.create(best_obs)
                    except Exception as e:
                        self._pg_available = False
                        logger.warning(f"PostgreSQL offline/unreachable — skipping DB evidence persistence for subsequent tracks: {e}")
            else:
                logger.warning(f"No quality crops found for track {track_id}")

        # 7. Automatic Video Vision & Metadata Extraction Engine
        try:
            from app.cv.metadata.extractor import AutoVideoMetadataExtractor
            from pymongo import MongoClient
            from app.config.db import db_settings

            extractor = AutoVideoMetadataExtractor()
            video_meta_doc = extractor.generate_video_metadata_json(
                video_id=video_id,
                camera_id=camera_id,
                track_obs_map=track_obs,
                resolved_pids=resolved_pids
            )
            logger.info(f"IngestionManager: Generated auto metadata JSON with {len(video_meta_doc.get('tracks', []))} tracks.")

            # Update PostgreSQL evidence records with auto-generated descriptions & attributes
            try:
                import asyncpg
                pg_uri = db_settings.POSTGRES_URI.replace('postgresql+asyncpg://', 'postgresql://')
                pg_conn = await asyncpg.connect(pg_uri)
                for t_meta in video_meta_doc.get("tracks", []):
                    t_id = t_meta["track_id"]
                    desc = t_meta["description"]
                    attr_json = json.dumps({
                        "gender": t_meta.get("gender"),
                        "behavior": t_meta.get("behavior"),
                        "location": t_meta.get("location"),
                        "spatial_zone": t_meta.get("spatial_zone"),
                        "crop_url": t_meta.get("crop_url")
                    })
                    await pg_conn.execute('''
                        UPDATE evidence
                        SET description = $1, attributes = $2
                        WHERE track_uuid IN (
                            SELECT id FROM tracks WHERE track_id = $3
                        )
                    ''', desc, attr_json, t_id)
                await pg_conn.close()
            except Exception as pg_err:
                logger.warning(f"PostgreSQL metadata enrichment update failed: {pg_err}")

            # Update MongoDB observations and persist events
            if getattr(self, "_mongo_available", True):
                try:
                    mc = MongoClient(db_settings.MONGO_URI, serverSelectionTimeoutMS=1000)
                    mc.admin.command('ping')
                    db = mc[db_settings.MONGO_DB_NAME]
                    for t_meta in video_meta_doc.get("tracks", []):
                        t_id = t_meta["track_id"]
                        db["observations"].update_many(
                            {"original_track_id": t_id},
                            {"$set": {
                                "description": t_meta["description"],
                                "behavior": t_meta.get("behavior"),
                                "gender": t_meta.get("gender"),
                                "location": t_meta.get("location"),
                                "spatial_zone": t_meta.get("spatial_zone"),
                                "crop_url": t_meta.get("crop_url")
                            }}
                        )

                    events_list = video_meta_doc.get("events", [])
                    if events_list:
                        db["events"].insert_many(events_list)
                    mc.close()
                except Exception as m_err:
                    logger.warning(f"MongoDB metadata enrichment update failed: {m_err}")

        except Exception as meta_err:
            logger.error(f"Automatic video metadata extraction failed: {meta_err}")

        logger.info(f"IngestionManager: Completed processing and metadata generation for {video_id}")
        return True
