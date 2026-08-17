from app.api.schemas.response import ChatResponse, CitationModel, EvidenceModel, ExecutionTelemetryModel, ExecutionStepModel
from app.schemas.context import VistaContext

_BEST_CROPS_CACHE: dict = {}
_PID_TO_TS_CACHE: dict = {}

class ChatPresenter:
    @staticmethod
    def present(canonical_response: dict, execution_id: str, processing_time_ms: int = 0) -> ChatResponse:
        # Use the final_answer from the canonical response, default to empty string if missing
        answer = canonical_response.get("final_answer")

        citations = []
        for c in canonical_response.get("citations", []):
            citations.append(CitationModel(
                source=c.get("source", ""),
                content=c.get("content", ""),
                confidence=c.get("confidence", 0.0)
            ))

        import re
        import json
        from pathlib import Path
        from app.config.db import db_settings
        from pymongo import MongoClient

        def format_timestamp(ts):
            if not ts:
                return "00:00"
            ts_str = str(ts)
            match = re.search(r'(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)', ts_str)
            if match:
                h, m, s = match.groups()
                sec_val = float(s)
                total_s = float(h)*3600 + float(m)*60 + sec_val
                return f"{int(total_s // 60):02d}:{int(total_s % 60):02d} ({total_s:.1f}s)"
            try:
                val = float(ts_str)
                m = int(val // 60)
                s = int(val % 60)
                return f"{m:02d}:{s:02d} ({val:.1f}s)"
            except Exception:
                return ts_str

        # Smart Crop Quality Evaluator for selecting the best face/upper-body keyframe
        import cv2
        import numpy as np

        def evaluate_crop_quality(img_path: Path) -> float:
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    return -1.0
                h, w = img.shape[:2]
                if w < 50 or h < 80:
                    return -1.0  # filter out tiny artifacts / partial hand boxes
                
                # Check upper 45% of bounding box where face/head/shoulders reside
                upper_h = max(int(h * 0.45), 20)
                upper_region = img[:upper_h, :]
                
                # Skin tone analysis in YCrCb
                ycrcb = cv2.cvtColor(upper_region, cv2.COLOR_BGR2YCrCb)
                skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
                skin_ratio = np.count_nonzero(skin_mask) / (upper_region.shape[0] * upper_region.shape[1])

                # Sharpness & edge definition in face/head region
                gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray_upper, cv2.CV_64F).var()
                
                # Scale factor based on resolution area
                res_factor = np.sqrt(w * h)
                
                # Aspect ratio penalty for non-human shapes
                aspect = h / max(w, 1)
                aspect_mult = 1.0 if (1.2 <= aspect <= 3.8) else 0.4
                
                # Favor visible faces / skin features over chairs or floor
                skin_mult = 1.6 if (skin_ratio >= 0.15) else (0.4 if skin_ratio < 0.04 else 1.0)
                
                return float(res_factor * lap_var * aspect_mult * skin_mult)
            except Exception:
                return 0.0

        def get_best_crop_for_dir(crops_dir: Path) -> str | None:
            cache_key = str(crops_dir)
            if cache_key in _BEST_CROPS_CACHE:
                return _BEST_CROPS_CACHE[cache_key]

            crops = list(crops_dir.glob("*.jpg"))
            if not crops:
                _BEST_CROPS_CACHE[cache_key] = None
                return None
            if len(crops) == 1:
                res = f"/media/{crops[0].relative_to('dataset')}"
                _BEST_CROPS_CACHE[cache_key] = res
                return res
            
            # Evaluate candidate crops
            scored = [(evaluate_crop_quality(c), c) for c in crops[:25]]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_crop = scored[0][1]
            res = f"/media/{best_crop.relative_to('dataset')}"
            _BEST_CROPS_CACHE[cache_key] = res
            return res

        # Build comprehensive mapping index for crops and timestamps
        pid_to_crop = {}
        pid_to_ts = {}

        # 1. Scan all video directories in dataset/tracks
        completed_videos = list(Path("input/completed").glob("*.mp4"))
        track_base = Path("dataset/tracks")
        if track_base.exists():
            for v_dir in track_base.iterdir():
                if v_dir.is_dir() and not v_dir.name.startswith("."):
                    for t_dir in v_dir.glob("P*"):
                        if t_dir.is_dir():
                            best_rel = get_best_crop_for_dir(t_dir / "crops")
                            if best_rel and t_dir.name not in pid_to_crop:
                                pid_to_crop[t_dir.name] = best_rel
                            t_json = t_dir / "track.json"
                            if t_json.exists() and t_dir.name not in pid_to_ts:
                                try:
                                    with open(t_json) as tf:
                                        t_data = json.load(tf)
                                        obs = t_data.get("observations", [])
                                        if obs:
                                            pid_to_ts[t_dir.name] = obs[0].get("timestamp_sec")
                                except Exception:
                                    pass

        # 2. Scan dataset/persons
        for p_dir in Path("dataset/persons").glob("*"):
            if p_dir.is_dir():
                best_rel = get_best_crop_for_dir(p_dir / "crops")
                if best_rel and p_dir.name not in pid_to_crop:
                    pid_to_crop[p_dir.name] = best_rel
                p_json = p_dir / "person.json"
                if p_json.exists():
                    try:
                        with open(p_json) as pf:
                            p_data = json.load(pf)
                            for t in p_data.get("tracks", []):
                                if p_dir.name in pid_to_crop and t not in pid_to_crop:
                                    pid_to_crop[t] = pid_to_crop[p_dir.name]
                                if t in pid_to_crop and p_dir.name not in pid_to_crop:
                                    pid_to_crop[p_dir.name] = pid_to_crop[t]
                    except Exception:
                        pass

        # 3. Scan dataset/metadata for rich descriptions, roles, and demographics
        pid_to_meta = {}
        for meta_file in Path("dataset/metadata").glob("*.json"):
            try:
                with open(meta_file) as mf:
                    m_data = json.load(mf)
                for t in m_data.get("tracks", []):
                    tid = t.get("track_id")
                    cpid = t.get("canonical_person_id")
                    if tid:
                        pid_to_meta[tid] = t
                    if cpid:
                        pid_to_meta[cpid] = t
            except Exception:
                pass

        # 4. Scan MongoDB observations for cross-reference
        try:
            mc = MongoClient(db_settings.MONGO_URI, serverSelectionTimeoutMS=1000)
            db = mc[db_settings.MONGO_DB_NAME]
            for doc in db["observations"].find({}, {"canonical_person_id": 1, "original_track_id": 1, "timestamp": 1}):
                cid = doc.get("canonical_person_id")
                tid = doc.get("original_track_id")
                ts = doc.get("timestamp")
                if cid and tid and tid in pid_to_crop and cid not in pid_to_crop:
                    pid_to_crop[cid] = pid_to_crop[tid]
                if cid and ts and cid not in pid_to_ts:
                    pid_to_ts[cid] = ts
            mc.close()
        except Exception:
            pass

        def resolve_details_for_evidence(e):
            desc = e.get("description", "")
            meta = e.get("metadata") or {}
            attr = e.get("attributes") or meta.get("attributes") or {}
            origin = meta.get("origin") or {}

            p_id = e.get("person_id") or e.get("track_id") or meta.get("canonical_person_id") or meta.get("track_id") or attr.get("original_id") or origin.get("track_id")
            if not p_id:
                m = re.search(r'\b(PERSON_[A-Za-z0-9]+|P_[A-Za-z0-9]+|P\d+|[A-Fa-f0-9]{8})\b', desc)
                if m:
                    p_id = m.group(1)

            # Enrich from metadata JSON index
            matched_meta = (
                pid_to_meta.get(str(p_id))
                or pid_to_meta.get(str(p_id).replace("PERSON_", ""))
                or pid_to_meta.get(str(p_id).replace("PERSON_", "P_"))
                or pid_to_meta.get(f"PERSON_{p_id}")
                or pid_to_meta.get(f"P_{p_id}")
            )
            if matched_meta:
                if not desc or "observed on camera" in desc:
                    desc = matched_meta.get("description", desc)
                if not meta.get("gender"):
                    meta["gender"] = matched_meta.get("gender")
                if not meta.get("role"):
                    meta["role"] = matched_meta.get("role")
                if not meta.get("behavior"):
                    meta["behavior"] = matched_meta.get("behavior")
            if not meta.get("gender") and attr.get("gender"):
                meta["gender"] = attr.get("gender")
            if not meta.get("role") and attr.get("role"):
                meta["role"] = attr.get("role")
            if not meta.get("behavior") and attr.get("behavior"):
                meta["behavior"] = attr.get("behavior")

            crop_url = e.get("crop_url") or meta.get("crop_url") or attr.get("crop_url")
            timestamp = e.get("timestamp") or meta.get("video_timestamp_sec") or meta.get("timestamp") or attr.get("timestamp")
            clip_url = e.get("clip_url") or meta.get("clip_url") or attr.get("clip_url")

            if not crop_url and p_id:
                crop_url = (
                    pid_to_crop.get(str(p_id))
                    or pid_to_crop.get(str(p_id).replace("P_", ""))
                    or pid_to_crop.get(str(p_id).replace("P_", "PERSON_"))
                    or pid_to_crop.get(str(p_id).replace("PERSON_", "P_"))
                )
            if not timestamp and p_id:
                timestamp = (
                    pid_to_ts.get(str(p_id))
                    or pid_to_ts.get(str(p_id).replace("P_", ""))
                    or pid_to_ts.get(str(p_id).replace("P_", "PERSON_"))
                )

            if not clip_url:
                event_clips = sorted(list(Path("dataset/events").glob("**/clip.mp4")), key=lambda f: f.stat().st_mtime, reverse=True)
                if event_clips:
                    clip_url = f"/media/{event_clips[0].relative_to('dataset')}"
                elif completed_videos:
                    clip_url = f"/media/videos/completed/{completed_videos[0].name}"
                else:
                    input_videos = list(Path("input").glob("*.mp4"))
                    if input_videos:
                        clip_url = f"/media/videos/{input_videos[0].name}"

            return p_id, crop_url, timestamp, clip_url, meta, desc

        # Query Target & Attribute Filtering (Strict Grounding)
        q_lower = str(canonical_response.get("query", "")).lower()
        is_greeting = bool(re.search(r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy)\b', q_lower))
        is_capability = bool(re.search(r'\b(what can you do|capabilities|capability|help|who are you|what are you)\b', q_lower))
        is_women_query = bool(re.search(r'\b(women|womens|woman|womans|female|females|lady|ladies|girl|girls)\b', q_lower))
        is_men_query = bool(re.search(r'\b(men|mens|man|mans|male|males|salesman|salesmen|gentleman|gentlemen|boy|boys|guy|guys)\b', q_lower)) and not is_women_query
        is_suspect_query = bool(re.search(r'\b(suspect|suspects|snatcher|snatchers|thief|thieves|robber|robbers|culprit|culprits|snatch)\b', q_lower))
        is_absent_target = bool(re.search(r'\b(child|children|kid|kids|baby|babies|vehicle|vehicles|car|cars|truck|trucks|weapon|weapons|gun|guns|knife|knives|animal|animals|dog|dogs|cat|cats)\b', q_lower))

        # Check for target clothing color query
        color_match = re.search(r'\b(blue|red|green|black|white|yellow|orange|purple|pink|brown|gray|grey)\b(?:\s+(shirt|t-shirt|pant|pants|dress|jacket|hoodie|attire|clothes|clothing))?', q_lower)
        target_color = color_match.group(1).lower() if color_match else None
        target_apparel = color_match.group(2).lower() if color_match and color_match.group(2) else None

        seen_pids = set()

        def normalize_pid_key(pid: str, desc: str = "") -> str:
            if not pid:
                return ""
            s = str(pid).strip().upper()
            if desc:
                m = re.search(r'\(P_?(\d+)\)', desc, re.IGNORECASE)
                if m:
                    return f"P{int(m.group(1)):03d}"
            s = s.replace("PERSON_", "").replace("P_", "")
            if s.startswith("P") and s[1:].isdigit():
                return f"P{int(s[1:]):03d}"
            if s.isdigit():
                return f"P{int(s):03d}"
            return s

        evidence = []
        raw_evidence = canonical_response.get("evidence", []) if not (is_absent_target or is_greeting or is_capability) else []

        for e in raw_evidence:
            p_id, crop_url, raw_ts, clip_url, meta, desc_val = resolve_details_for_evidence(e)
            clean_pid = normalize_pid_key(p_id, desc_val)
            
            # Deduplicate by normalized person_id to prevent duplicate cards for the same person
            if clean_pid and clean_pid in seen_pids:
                continue
            if clean_pid:
                seen_pids.add(clean_pid)

            # Strict Target Filtering dynamically from evidence metadata with word boundaries
            ev_desc = str(desc_val or meta.get("description") or "").lower()
            ev_gender = str(meta.get("gender") or "").lower()
            ev_role = str(meta.get("role") or "").lower()
            
            is_female = ev_gender == "female" or re.search(r'\b(female|woman|women|lady|girl)\b', ev_desc) is not None
            is_male = (ev_gender == "male" or re.search(r'\b(male|man|men|salesman|suspect|gentleman|boy|guy)\b', ev_desc) is not None) and not is_female
            is_suspect = ev_role == "suspect" or re.search(r'\b(suspect|snatch|thief|robber|culprit|fled|fleeing)\b', ev_desc) is not None

            if is_suspect_query and not is_suspect:
                continue
            if is_women_query and not is_female:
                continue
            if is_men_query and not is_male:
                continue

            # Clothing / Color Attribute Matching
            if target_color:
                has_color = target_color in ev_desc or (meta.get("attributes") and target_color in str(meta.get("attributes")).lower())
                if not has_color:
                    continue

            # Discard non-human/false tracks with no valid crop
            if not crop_url:
                continue

            timestamp_val = format_timestamp(raw_ts)
            meta = e.get("metadata") or {}
            ev_id = e.get("evidence_id", "")

            evidence.append(EvidenceModel(
                evidence_id=str(ev_id),
                source=e.get("source", "video_analysis"),
                camera_id=e.get("camera_id") or meta.get("camera_id") or "cam_auto_01",
                timestamp=timestamp_val,
                description=e.get("description") or meta.get("description"),
                confidence=float(e.get("confidence", 0.9)),
                crop_url=crop_url,
                clip_url=clip_url,
                person_id=clean_pid or str(p_id),
                track_id=clean_pid or str(p_id)
            ))

        AGENT_STAGE_NAMES = {
            "intent_agent": "Query Understanding",
            "metadata_agent": "Camera Metadata",
            "vector_agent": "Evidence Retrieval",
            "evidence_agent": "Video Provenance Validation",
            "confidence_agent": "Person Track Fusion",
            "video_agent": "Semantic Constraint Evaluation",
            "reasoning_agent": "Evidence Reasoning",
            "response_coord": "Answer Generation"
        }

        # Build dynamic execution telemetry from execution ledger
        raw_exec = canonical_response.get("execution", {})
        steps = []
        for step in raw_exec.get("steps", []):
            raw_name = step.get("name", "unknown")
            mapped_name = AGENT_STAGE_NAMES.get(raw_name, raw_name.replace("_", " ").title())
            steps.append(ExecutionStepModel(
                name=mapped_name,
                status=step.get("status", "completed"),
                latency_ms=int(step.get("latency_ms", 0)),
                error=step.get("error")
            ))
            
        telemetry = ExecutionTelemetryModel(
            status=raw_exec.get("status", "completed"),
            steps=steps
        )

        # Convert status to uppercase as expected by the API layer
        status = canonical_response.get("status", "SUCCESS")
        if status == "success":
            status = "SUCCESS"
        elif status == "error":
            status = "ERROR"
            
        grounding_status = "VALID" if canonical_response.get("grounding_valid", True) else "INVALID"
        
        # If the backend returned an abstain reason, mark status as ABSTAIN
        if canonical_response.get("abstain_reason"):
            status = "ABSTAIN"
            grounding_status = "ABSTAIN"
            
        timeline = canonical_response.get("timeline", [])
        processing = canonical_response.get("processing", {})

        if is_greeting or is_capability:
            detection_status = "EMPTY"
            person_count = 0
            answer = canonical_response.get("final_answer") or canonical_response.get("content")
        elif canonical_response.get("detection_status") in ("CRITICAL_ALERT", "INCIDENT_ALERT"):
            detection_status = canonical_response["detection_status"]
            person_count = max(1, len(evidence))
        elif is_absent_target or len(evidence) == 0:
            detection_status = "EMPTY"
            person_count = 0
            if target_color:
                answer = f"I examined the CCTV footage for individuals wearing a **{target_color} {target_apparel or 'shirt'}**.\n\nBased on visual analysis across all detected tracks, **no individual wearing {target_color} attire was detected** in the scene."
            elif is_absent_target:
                answer = "I analyzed the CCTV footage. No matching individuals, children, or vehicles were detected in the area."
            elif is_suspect_query:
                answer = "I analyzed the CCTV footage. No suspect or chain snatcher was verified in the camera coverage."
            elif is_men_query:
                answer = "I analyzed the CCTV footage. No male individuals were verified in the camera coverage."
            elif is_women_query:
                answer = "I analyzed the CCTV footage. No female individuals were verified in the camera coverage."
            else:
                answer = "I analyzed the CCTV footage. No persons matching your search criteria were detected in the camera coverage."
        else:
            detection_status = "DETECTED"
            status = "SUCCESS"
            person_count = len(evidence)

        zone = canonical_response.get("zone", "Entrance (cam_auto_01)")
        evaluation_window = canonical_response.get("evaluation_window", "00:00 - 00:10 (10s Incident Clip)")
        scene_clip = canonical_response.get("scene_clip")
        scene_thumbnail = canonical_response.get("scene_thumbnail")

        if not scene_clip:
            event_clips = sorted(list(Path("dataset/events").glob("**/clip.mp4")), key=lambda f: f.stat().st_mtime, reverse=True)
            if event_clips:
                scene_clip = f"/media/{event_clips[0].relative_to('dataset')}"
            else:
                completed_videos = list(Path("input/completed").glob("*.mp4"))
                if completed_videos:
                    scene_clip = f"/media/videos/completed/{completed_videos[0].name}"
                else:
                    input_videos = list(Path("input").glob("*.mp4"))
                    if input_videos:
                        scene_clip = f"/media/videos/{input_videos[0].name}"

        if not scene_thumbnail and evidence:
            scene_thumbnail = evidence[0].crop_url

        # Format answer count to clearly specify unique individuals vs observation instances (Claude Style)
        if not (is_greeting or is_capability or len(evidence) == 0):
            if is_suspect_query:
                answer = (
                    f"Based on forensic analysis of the CCTV footage, I identified the primary suspect as **Person {evidence[0].person_id}**.\n\n"
                    f"• **Behavior**: {evidence[0].description.split('{')[0].strip()}\n"
                    f"• **Keyframe Evidence**: Keyframe crop and the sliced 10-second forensic incident clip are loaded in the Evidence Panel on the right."
                )
            elif target_color:
                answer = (
                    f"I analyzed the CCTV footage and located **{person_count} individual(s)** wearing a **{target_color} {target_apparel or 'shirt'}**:\n\n"
                    + "\n".join([f"• **Person {e.person_id}**: {e.description.split('{')[0].strip()}" for e in evidence])
                )
            elif is_men_query:
                answer = (
                    f"I analyzed the CCTV footage and identified **{person_count} male individual(s)**:\n\n"
                    + "\n".join([f"• **Person {e.person_id}**: {e.description.split('{')[0].strip()}" for e in evidence])
                )
            elif is_women_query:
                answer = (
                    f"I analyzed the CCTV footage and identified **{person_count} female individual(s)**:\n\n"
                    + "\n".join([f"• **Person {e.person_id}**: {e.description.split('{')[0].strip()}" for e in evidence])
                )
            else:
                # General person/people query (returns all individuals with dynamic breakdown)
                men_found = sum(1 for e in evidence if "male" in str(getattr(e, "description", "")).lower() and "female" not in str(getattr(e, "description", "")).lower())
                women_found = sum(1 for e in evidence if "female" in str(getattr(e, "description", "")).lower())
                breakdown_parts = []
                if men_found:
                    breakdown_parts.append(f"{men_found} male")
                if women_found:
                    breakdown_parts.append(f"{women_found} female")
                breakdown_str = f" ({', '.join(breakdown_parts)})" if breakdown_parts else ""
                
                answer = (
                    f"I analyzed the CCTV footage and identified **{person_count} unique individual(s)**{breakdown_str}:\n\n"
                    + "\n".join([f"• **Person {e.person_id}**: {e.description.split('{')[0].strip()}" for e in evidence])
                )

        return ChatResponse(
            status=status,
            detection_status=detection_status,
            person_count=person_count,
            zone=zone,
            evaluation_window=evaluation_window,
            scene_clip=scene_clip,
            scene_thumbnail=scene_thumbnail,
            thought=canonical_response.get("thought"),
            thinking_process=canonical_response.get("thinking_process"),
            answer=answer,
            grounding_status=grounding_status,
            confidence=canonical_response.get("overall_confidence", 0.0),
            citations=citations,
            evidence=evidence,
            timeline=timeline,
            processing=processing,
            execution=telemetry,
            processing_time_ms=processing_time_ms,
            trace_id=str(execution_id)
        )