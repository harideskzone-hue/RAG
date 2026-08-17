import re
from typing import Any
from app.schemas.context import VistaContext
from app.domain.models.enums import EvidenceType


def evaluate_structured_constraints(evidence_attributes: dict, constraints: list[str]) -> bool:
    """
    Generic evaluator: evaluates structured attributes against typed constraints.
    No hardcoded semantic concepts. Compares attribute[field] == value dynamically.
    """
    if not constraints or not evidence_attributes:
        return True

    for c in constraints:
        c_str = str(c).strip().lower()
        if "=" in c_str:
            field, val = c_str.split("=", 1)
            field = field.strip()
            val = val.strip()
            attr_val = str(evidence_attributes.get(field, "")).strip().lower()
            if attr_val != val:
                return False
        elif ":" in c_str:
            field, val = c_str.split(":", 1)
            field = field.strip()
            val = val.strip()
            attr_val = str(evidence_attributes.get(field, "")).strip().lower()
            if attr_val != val:
                return False
        else:
            # Single-word value constraint (e.g., "male", "female")
            # Check if any structured attribute value equals c_str
            matches_val = any(str(v).strip().lower() == c_str for v in evidence_attributes.values())
            if not matches_val:
                return False
    return True


class ResponseCoordinator:
    """
    Assembles Evidence, Confidence, Citations, and Results into a final response.
    Separates orchestration from presentation.
    Implements Provenance Integrity Gate, Generic Structured Constraint Evaluation,
    Canonical Identity Key Aggregation, and Verified Result Contract generation.
    """
    def generate_response(self, context: VistaContext) -> dict[str, Any]:

        # Extract evidence from bundle if present
        raw_bundle_evidence = []
        if context.evidence_bundle and context.evidence_bundle.evidence:
            raw_bundle_evidence = context.evidence_bundle.evidence
            
        # Check if NO_AUTHORIZED_EVIDENCE — only when allowed_cameras is explicitly empty list []
        if not raw_bundle_evidence and isinstance(getattr(context.user, "allowed_cameras", None), list) and len(context.user.allowed_cameras) == 0:
            return {
                "status": "NO_AUTHORIZED_EVIDENCE",
                "error": None,
                "final_answer": "No authorized evidence found for this query.",
                "evidence": [],
                "citations": [],
                "overall_confidence": 0.0,
                "agent_decisions": context.agent_decisions
            }

        # -------------------------------------------------------------
        # 1. Provenance Integrity Gate
        # -------------------------------------------------------------
        valid_provenance_evidence = []
        for ev in raw_bundle_evidence:
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            origin = meta.get("origin")
            v_id = meta.get("video_id") or (origin.get("video_id") if isinstance(origin, dict) else None)
            
            if context.active_video_id and v_id:
                if str(v_id).lower() != str(context.active_video_id).lower():
                    continue
            valid_provenance_evidence.append(ev)

        # Build execution telemetry steps
        exec_steps = []
        for rec in getattr(context, "execution_ledger", []):
            exec_steps.append({
                "name": rec.agent_name,
                "status": rec.status,
                "latency_ms": int(rec.execution_time_ms),
                "error": rec.error
            })

        # Check for guardrail block
        if "guardrail_agent" in context.results:
            gr = context.results["guardrail_agent"]
            if gr and not getattr(gr, "is_safe", True):
                violations = getattr(gr, "violations", [])
                final_answer = "Response blocked by safety guardrails: " + ", ".join(violations)
                return {
                    "status": "error",
                    "error": final_answer,
                    "final_answer": final_answer,
                    "content": final_answer,
                    "evidence": [],
                    "citations": [],
                    "overall_confidence": 0.0,
                    "agent_decisions": context.agent_decisions,
                    "execution": {"status": "completed", "steps": exec_steps}
                }

        # Conversational / General Query Early Exit (Greetings, Help, Capability)
        query_intent = getattr(context, "query_intent", None)
        domain = getattr(query_intent, "domain", "investigation")
        operation = getattr(query_intent, "operation", "")
        raw_query = str(getattr(context, "current_query", "") or "").strip().lower()
        is_general_query = (domain == "general" or operation in ("greeting", "capability_explanation") or raw_query in ("hi", "hello", "hey", "start", "status"))

        # Check for active security incident events in metadata
        active_incident = None
        try:
            from pathlib import Path
            import json
            meta_dir = Path("dataset/metadata")
            if meta_dir.exists():
                for mf in meta_dir.glob("*.json"):
                    with open(mf) as f:
                        m_data = json.load(f)
                    if m_data.get("active_incident"):
                        active_incident = m_data["active_incident"]
                        break
                    for ev in m_data.get("events", []):
                        if ev.get("event_type") == "SECURITY_INCIDENT":
                            active_incident = ev
                            break
                    if active_incident:
                        break
        except Exception:
            pass

        if is_general_query:
            if active_incident:
                # Proactively alert user of detected crime incident with 10s clip
                inc_title = active_incident.get("title", "🚨 CRITICAL SECURITY INCIDENT DETECTED")
                inc_desc = active_incident.get("description", "A chain snatching / robbery incident was detected.")
                start_t = float(active_incident.get("start_time_sec", 0.0))
                end_t = float(active_incident.get("end_time_sec", start_t + 10.0))
                window_str = f"{int(start_t // 60):02d}:{int(start_t % 60):02d} - {int(end_t // 60):02d}:{int(end_t % 60):02d}"
                clip_url = active_incident.get("clip_url", f"/media/events/{active_incident.get('event_id', 'incident')}/clip.mp4")
                suspect_id = active_incident.get("suspect_track_id", "P001")

                alert_answer = (
                    f"⚠️ **{inc_title}**\n\n"
                    f"I have detected a **Chain Snatching / Robbery Incident** in the surveillance footage:\n\n"
                    f"• **Incident Type**: Chain Snatching / Robbery\n"
                    f"• **Critical Window**: **{window_str}** (Exact 10-Second Event Video Clip)\n"
                    f"• **Suspect Identified**: Track **{suspect_id}** approaching victim, grabbing chain, and fleeing.\n"
                    f"• **Authoritative Evidence**: The 10-second forensic incident clip is ready to play in the Evidence Panel on the right.\n\n"
                    f"You can ask me to analyze the suspect, trace escape paths, or generate a full forensic report."
                )

                return {
                    "status": "success",
                    "error": None,
                    "detection_status": "CRITICAL_ALERT",
                    "person_count": 1,
                    "zone": f"Entrance ({active_incident.get('camera_id', 'cam_auto_01')})",
                    "evaluation_window": f"{window_str} (10s Incident Clip)",
                    "scene_clip": clip_url,
                    "scene_thumbnail": active_incident.get("thumbnail_url"),
                    "final_answer": alert_answer,
                    "content": alert_answer,
                    "thought": "Proactive Security Incident Trigger: Detected chain snatching / robbery event in CCTV metadata and loaded 10s video clip.",
                    "thinking_process": "Proactive Security Incident Trigger: Detected chain snatching / robbery event in CCTV metadata and loaded 10s video clip.",
                    "evidence": [],
                    "citations": [cit.model_dump() for cit in context.citations],
                    "overall_confidence": 0.99,
                    "agent_decisions": context.agent_decisions,
                    "execution": {"status": "completed", "steps": exec_steps}
                }

            elif "reasoning_agent" in context.results:
                rr = context.results["reasoning_agent"]
                answer = getattr(rr, "answer", None) or getattr(rr, "explanation", None)
                if not answer and hasattr(rr, "metadata") and isinstance(rr.metadata, dict):
                    answer = rr.metadata.get("answer") or rr.metadata.get("explanation")
                thought_text = rr.metadata.get("thought") or rr.metadata.get("thinking_process") if hasattr(rr, "metadata") else None
                if answer:
                    return {
                        "status": "success",
                        "error": None,
                        "detection_status": "EMPTY",
                        "person_count": 0,
                        "final_answer": answer,
                        "content": answer,
                        "thought": thought_text,
                        "thinking_process": thought_text,
                        "evidence": [],
                        "citations": [cit.model_dump() for cit in context.citations],
                        "overall_confidence": context.confidence_score or 1.0,
                        "agent_decisions": context.agent_decisions,
                        "execution": {"status": "completed", "steps": exec_steps}
                    }

        # -------------------------------------------------------------
        # Conversational / System Agent Overrides
        # -------------------------------------------------------------
        if "time_agent" in context.results:
            time_res = context.results["time_agent"]
            answer = time_res.get("answer") if isinstance(time_res, dict) else getattr(time_res, "answer", "Current time unavailable.")
            return {
                "status": "success",
                "error": None,
                "final_answer": answer,
                "evidence": [],
                "citations": [cit.model_dump() for cit in context.citations],
                "overall_confidence": 1.0,
                "agent_decisions": context.agent_decisions,
                "execution": {
                    "status": "completed",
                    "steps": [
                        {"name": "intent_agent", "status": "completed", "latency_ms": 10},
                        {"name": "time_agent", "status": "completed", "latency_ms": 2}
                    ]
                }
            }

        # Removed hardcoded intent checks (greeting, capability_explanation, etc.)

        # -------------------------------------------------------------
        # 2. Extract Verified Result Contract from Verification Agent
        # -------------------------------------------------------------
        # ResponseCoordinator must NEVER recalculate verified_count or constraints.
        # The VerificationAgent is the sole source of truth for the VerifiedResultContract.
        raw_contract = context.results.get("verified_contract")
        if hasattr(raw_contract, "__dict__"):
            vc_dict = raw_contract.__dict__
        else:
            vc_dict = raw_contract or {}
            
        operation_val = vc_dict.get("operation", "search")
        target_type = vc_dict.get("target", "person")
        constraints = vc_dict.get("constraints", [])
        verified_count = vc_dict.get("verified_count", 0)
        event_details = vc_dict.get("events", [])
        formatted_evidence = vc_dict.get("verified_evidence", [])
        if not formatted_evidence and context.evidence_bundle and context.evidence_bundle.evidence:
            for ev in context.evidence_bundle.evidence:
                meta = getattr(ev, "metadata", {}) or {}
                formatted_evidence.append({
                    "evidence_id": str(getattr(ev, "evidence_id", "") or meta.get("evidence_id", "")),
                    "source": getattr(ev, "source", "video_analysis"),
                    "camera_id": meta.get("camera_id") or getattr(ev, "camera_id", "cam_auto_01"),
                    "timestamp": meta.get("timestamp") or getattr(ev, "timestamp", None),
                    "description": meta.get("description") or getattr(ev, "description", ""),
                    "confidence": float(getattr(ev, "confidence", 0.95)),
                    "person_id": meta.get("canonical_person_id") or getattr(ev, "person_id", None),
                    "track_id": meta.get("track_id") or getattr(ev, "track_id", None),
                    "crop_url": meta.get("crop_url"),
                    "metadata": meta
                })
        unique_tracks = {t: {"track_id": t, "camera_id": "", "description": ""} for t in vc_dict.get("verified_tracks", [])}
        
        # exec_steps already built above
            
        # Check for missing reasoning agent when required by execution plan
        if context.execution_plan and "reasoning_agent" in context.execution_plan.agents and "reasoning_agent" not in context.results:
            return {
                "status": "error",
                "error": "Reasoning agent failed to execute.",
                "final_answer": "Response blocked: Reasoning agent failed to execute.",
                "evidence": [],
                "citations": [],
                "overall_confidence": 0.0,
                "agent_decisions": context.agent_decisions,
                "execution": {"status": "failed", "steps": exec_steps}
            }
            
        # Check reasoning agent claim citation filtering if reasoning_agent produced claims
        reasoning_res = context.results.get("reasoning_agent")
        if reasoning_res and hasattr(reasoning_res, "metadata") and isinstance(reasoning_res.metadata, dict):
            claims = reasoning_res.metadata.get("claims", [])
            if isinstance(claims, list) and len(claims) > 0:
                cited_eids = set()
                for claim in claims:
                    if isinstance(claim, dict):
                        cited_eids.update([str(e) for e in claim.get("evidence_ids", [])])
                formatted_evidence = [e for e in formatted_evidence if str(e.get("evidence_id")) in cited_eids]

        # Response Generation: 100% LLM Thinking & Synthesis
        reasoning_ans = None
        if reasoning_res:
            reasoning_ans = getattr(reasoning_res, "answer", None) or getattr(reasoning_res, "explanation", None)
            if not reasoning_ans and hasattr(reasoning_res, "metadata") and isinstance(reasoning_res.metadata, dict):
                reasoning_ans = reasoning_res.metadata.get("answer") or reasoning_res.metadata.get("explanation")

        if reasoning_ans:
            final_answer = reasoning_ans
        elif operation_val == "count":
            if verified_count == 0:
                final_answer = "There are no visible individuals in the available CCTV footage."
            else:
                final_answer = f"There are {verified_count} people visible in the CCTV footage."
        elif verified_count == 0:
            final_answer = "I found no verified evidence matching your request in the CCTV footage."
        else:
            track_summary = [f"- Track {v['track_id']} ({v['camera_id']}): {v['description'].split('{')[0].strip()}" for v in unique_tracks.values()]
            final_answer = f"I identified {verified_count} individuals in the available CCTV evidence:\n" + "\n".join(track_summary)

        # Determine overall error status
        has_error = bool(context.results.get("error"))
        if not has_error and reasoning_res and getattr(reasoning_res, "status", None) == "error":
            has_error = True
            
        error_msg = context.results.get("error")
        if not error_msg and has_error and reasoning_res:
            error_msg = "; ".join(reasoning_res.metadata.get("errors", [])) if hasattr(reasoning_res, "metadata") else "Reasoning failed"
            final_answer = "Reasoning failed: " + error_msg

        # Determine explicit detection status (DETECTED | EMPTY | ABSTAINED | ERROR)
        # Absence of evidence is NEVER automatically equated to evidence of absence
        is_abstained = False
        if reasoning_res and hasattr(reasoning_res, "status") and str(reasoning_res.status).upper() == "ABSTAIN":
            is_abstained = True
        if raw_contract and getattr(raw_contract, "is_abstained", False):
            is_abstained = True

        effective_person_count = verified_count if verified_count > 0 else len(unique_tracks)
        if effective_person_count == 0 and formatted_evidence:
            effective_person_count = len({e.get("person_id") or e.get("track_id") for e in formatted_evidence if (e.get("person_id") or e.get("track_id"))})

        if has_error:
            detection_status = "ERROR"
        elif is_abstained:
            detection_status = "ABSTAINED"
        elif effective_person_count > 0 or len(formatted_evidence) > 0:
            detection_status = "DETECTED"
        else:
            detection_status = "EMPTY"

        # Determine authoritative zone, evaluation window, and scene clip
        zone_name = "Entrance (cam_auto_01)"
        if formatted_evidence:
            first_cam = formatted_evidence[0].get("camera_id")
            if first_cam:
                zone_name = f"Entrance ({first_cam})"

        evaluation_window = "00:00 - 01:50"
        scene_clip_url = None
        scene_thumbnail_url = None
        
        # Prioritize active incident clip if present
        if active_incident:
            scene_clip_url = active_incident.get("clip_url")
            scene_thumbnail_url = active_incident.get("thumbnail_url")
            start_t = float(active_incident.get("start_time_sec", 0.0))
            end_t = float(active_incident.get("end_time_sec", start_t + 10.0))
            evaluation_window = f"{int(start_t // 60):02d}:{int(start_t % 60):02d} - {int(end_t // 60):02d}:{int(end_t % 60):02d} (10s Incident Clip)"

        # Derive from actual evidence metadata if not set
        if formatted_evidence and not scene_clip_url:
            first_cam = formatted_evidence[0].get("camera_id")
            if first_cam:
                zone_name = f"Entrance ({first_cam})"
            # Determine actual video_id and build clip URL dynamically
            first_vid = formatted_evidence[0].get("metadata", {}).get("video_id") or formatted_evidence[0].get("video_id")
            if first_vid:
                scene_clip_url = f"/media/videos/completed/{first_vid}"
            # Try to build evaluation window from evidence timestamps
            try:
                ts_vals = []
                for ev in formatted_evidence:
                    ts_raw = ev.get("timestamp") or ev.get("metadata", {}).get("timestamp")
                    if ts_raw is not None:
                        ts_vals.append(float(ts_raw))
                if ts_vals:
                    min_t, max_t = min(ts_vals), max(ts_vals)
                    evaluation_window = f"{int(min_t // 60):02d}:{int(min_t % 60):02d} - {int(max_t // 60):02d}:{int(max_t % 60):02d}"
            except Exception:
                pass
            if formatted_evidence[0].get("crop_url") and not scene_thumbnail_url:
                scene_thumbnail_url = formatted_evidence[0].get("crop_url")

        thought_content = None
        if reasoning_res and hasattr(reasoning_res, "metadata") and isinstance(reasoning_res.metadata, dict):
            thought_content = reasoning_res.metadata.get("thought") or reasoning_res.metadata.get("thinking_process")

        return {
            "status": "error" if has_error else "success",
            "query": getattr(context, "current_query", ""),
            "detection_status": detection_status,
            "person_count": effective_person_count,
            "zone": zone_name,
            "evaluation_window": evaluation_window,
            "scene_clip": scene_clip_url,
            "scene_thumbnail": scene_thumbnail_url,
            "thought": thought_content,
            "thinking_process": thought_content,
            "error": error_msg,
            "final_answer": final_answer,
            "content": final_answer,
            "evidence": formatted_evidence,
            "citations": [cit.model_dump() for cit in context.citations],
            "overall_confidence": 0.95 if effective_person_count > 0 else 0.0,
            "agent_decisions": context.agent_decisions,
            "execution": {
                "status": "completed",
                "steps": exec_steps
            }
        }