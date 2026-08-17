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
            
            # If active_video_id is set, strictly enforce video isolation and valid origin
            if context.active_video_id:
                if not origin or not isinstance(origin, dict):
                    continue
                source_type = str(origin.get("type", "")).lower()
                valid_sources = ["video_ingestion", "video_analysis", "event_query", "event_analysis", "video_agent", "event_agent"]
                if source_type not in valid_sources:
                    continue
                track_id = origin.get("track_id") or meta.get("track_id")
                if not track_id:
                    continue
                video_id = origin.get("video_id") or meta.get("video_id")
                if video_id and str(video_id).lower() != str(context.active_video_id).lower():
                    continue
            elif origin and isinstance(origin, dict):
                source_type = str(origin.get("type", "")).lower()
                valid_sources = ["video_ingestion", "video_analysis", "event_query", "event_analysis", "video_agent", "event_agent"]
                if source_type and source_type not in valid_sources:
                    continue
                    
            valid_provenance_evidence.append(ev)

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
                    "execution": {"status": "failed", "steps": []}
                }

        # Check for reasoning agent result when evidence bundle is empty or not provided
        if "reasoning_agent" in context.results and not valid_provenance_evidence:
            rr = context.results["reasoning_agent"]
            answer = getattr(rr, "answer", None) or getattr(rr, "explanation", None)
            if not answer and hasattr(rr, "metadata") and isinstance(rr.metadata, dict):
                answer = rr.metadata.get("answer") or rr.metadata.get("explanation")
            if answer:
                return {
                    "status": "success",
                    "error": None,
                    "final_answer": answer,
                    "content": answer,
                    "evidence": [],
                    "citations": [cit.model_dump() for cit in context.citations],
                    "overall_confidence": context.confidence_score or 0.85,
                    "agent_decisions": context.agent_decisions,
                    "execution": {"status": "completed", "steps": []}
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
        # 2. Structured Constraint Evaluation & Identity Key Aggregation
        # -------------------------------------------------------------
        intent_result = context.results.get("intent_agent")
        query_intent = getattr(intent_result, "query_intent", None) if intent_result else None
        
        operation_val = ""
        target_type = "person"
        constraints = []
        
        if query_intent:
            operation_val = str(getattr(query_intent, "operation", "")).lower()
            target_type = getattr(query_intent, "target_type", "person") or "person"
            sem_c = getattr(query_intent, "semantic_constraints", []) or []
            attr_c = getattr(query_intent, "attributes", []) or []
            constraints = list(set(sem_c + attr_c))
        # Removed generic intent fallback

        # Build execution telemetry steps
        exec_steps = []
        for rec in getattr(context, "execution_ledger", []):
            exec_steps.append({
                "name": rec.agent_name,
                "status": rec.status,
                "latency_ms": int(rec.execution_time_ms),
                "error": rec.error
            })

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

        # Process evidence items through Canonical Identity Key Aggregation
        unique_tracks = {}
        formatted_evidence = []
        
        is_behavioral_op = (operation_val in ["behavioral_investigation", "event_search"])
        
        for ev in valid_provenance_evidence:
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}
            attributes = meta.get("attributes") if isinstance(meta.get("attributes"), dict) else {}
            
            # Check generic structured constraints
            if not evaluate_structured_constraints(attributes, constraints):
                continue
            
            # Behavioral / Event evidence filter:
            # Behavioral investigation MUST operate on structured event/behavior evidence.
            # Person visual attributes (gender, clothing, hair, location) alone MUST NOT be treated as behavioral evidence.
            if is_behavioral_op:
                event_type = meta.get("event_type") or meta.get("behavior_type") or meta.get("anomaly_type") or origin.get("event_type")
                is_event_type = (getattr(ev, "evidence_type", None) == EvidenceType.EVENT) or bool(event_type)
                if not is_event_type:
                    continue
                
            # Canonical Identity Key: (video_id, camera_id, track_id)
            track_id = origin.get("track_id") or meta.get("track_id")
            if not track_id:
                if context.active_video_id:
                    continue
                track_id = str(ev.evidence_id)

            camera_id = origin.get("camera_id") or meta.get("camera_id") or ""
            video_id = origin.get("video_id") or context.active_video_id or ""
            
            identity_key = (video_id, camera_id, track_id)
            
            if identity_key not in unique_tracks:
                unique_tracks[identity_key] = {
                    "track_id": track_id,
                    "camera_id": camera_id,
                    "video_id": video_id,
                    "attributes": attributes,
                    "description": meta.get("description", ""),
                    "evidence": ev
                }
                
                # Format evidence card with real provenance
                source_label = origin.get("type", ev.source)
                if source_label in ["video_ingestion", "video_analysis"]:
                    source_label = "Video Analysis"
                    
                formatted_evidence.append({
                    "evidence_id": str(ev.evidence_id),
                    "source": source_label,
                    "camera_id": camera_id,
                    "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
                    "description": meta.get("description", ""),
                    "confidence": ev.confidence,
                    "origin": origin,
                    "attributes": attributes
                })

        # Check reasoning agent claim citation filtering if reasoning_agent produced claims
        reasoning_res = context.results.get("reasoning_agent")
        if reasoning_res and hasattr(reasoning_res, "metadata") and isinstance(reasoning_res.metadata, dict):
            claims = reasoning_res.metadata.get("claims", [])
            if isinstance(claims, list) and len(claims) > 0:
                cited_eids = set()
                for claim in claims:
                    if isinstance(claim, dict):
                        cited_eids.update([str(e) for e in claim.get("evidence_ids", [])])
                formatted_evidence = [e for e in formatted_evidence if e["evidence_id"] in cited_eids]

        # Verified Result Contract Construction
        verified_count = len(unique_tracks)
        track_ids = [v["track_id"] for v in unique_tracks.values()]
        
        # Extract event objects for behavioral contracts
        event_details = []
        if is_behavioral_op:
            for v in unique_tracks.values():
                ev_obj = v.get("evidence")
                meta_obj = ev_obj.metadata if hasattr(ev_obj, "metadata") and isinstance(ev_obj.metadata, dict) else {}
                origin_obj = meta_obj.get("origin") if isinstance(meta_obj.get("origin"), dict) else {}
                event_type_name = meta_obj.get("event_type") or meta_obj.get("behavior_type") or meta_obj.get("anomaly_type") or origin_obj.get("event_type") or "event"
                event_details.append({
                    "track_id": v["track_id"],
                    "camera_id": v["camera_id"],
                    "event_type": event_type_name,
                    "description": v["description"],
                    "video_timestamp_sec": origin_obj.get("video_timestamp_sec") or origin_obj.get("timestamp_sec")
                })
        
        verified_contract = {
            "status": "verified" if verified_count > 0 else "no_evidence",
            "operation": operation_val or "search",
            "target": target_type,
            "constraints": constraints,
            "verified_count": verified_count,
            "verified_tracks": track_ids,
            "events": event_details,
            "video_id": context.active_video_id or ""
        }
        context.results["verified_contract"] = verified_contract

        # Response Generation
        if operation_val == "count":
            if any("age_group" in str(c) or "child" in str(c) for c in constraints):
                final_answer = "Cannot verify the number of children from the available structured CCTV evidence."
            elif verified_count == 0:
                if constraints:
                    label = f"matching {target_type}s ({', '.join(constraints)})"
                else:
                    label = "people" if target_type == "person" else f"{target_type}s"
                final_answer = f"There are no visible {label} in the available CCTV footage."
            elif len(constraints) == 0:
                final_answer = f"There are {verified_count} people visible in the CCTV footage."
            else:
                label = f"matching {target_type}s ({', '.join(constraints)})"
                final_answer = f"There are {verified_count} {label} visible in the CCTV footage."
        elif operation_val in ["behavioral_investigation", "event_search"] or (context.current_query and any(w in context.current_query.lower() for w in ["suspicious", "fire", "accident", "explosion"])):
            if verified_count == 0:
                if context.current_query and "fire" in context.current_query.lower():
                    final_answer = "I couldn't verify any fire incidents or matching events in the available CCTV evidence."
                else:
                    final_answer = "I couldn't verify any suspicious activity or matching events in the available CCTV evidence."
            else:
                reasoning_ans = None
                if reasoning_res:
                    reasoning_ans = getattr(reasoning_res, "answer", None) or getattr(reasoning_res, "explanation", None)
                    if not reasoning_ans and hasattr(reasoning_res, "metadata") and isinstance(reasoning_res.metadata, dict):
                        reasoning_ans = reasoning_res.metadata.get("answer") or reasoning_res.metadata.get("explanation")
                if reasoning_ans:
                    final_answer = reasoning_ans
                else:
                    event_summary_parts = []
                    for ev_detail in event_details:
                        evt_name = str(ev_detail.get("event_type", "event")).replace("_", " ")
                        t_id = ev_detail.get("track_id")
                        ts_val = ev_detail.get("video_timestamp_sec")
                        ts_str = f" at {ts_val}s" if ts_val is not None else ""
                        event_summary_parts.append(f"Track {t_id} associated with a verified {evt_name} event{ts_str}")
                    
                    if event_summary_parts:
                        final_answer = f"I found {verified_count} person(s) associated with verified CCTV event(s): {'; '.join(event_summary_parts)}."
                    else:
                        final_answer = f"I found {verified_count} person(s) associated with verified CCTV event(s)."
        else:
            reasoning_ans = None
            if reasoning_res:
                reasoning_ans = getattr(reasoning_res, "answer", None) or getattr(reasoning_res, "explanation", None)
                if not reasoning_ans and hasattr(reasoning_res, "metadata") and isinstance(reasoning_res.metadata, dict):
                    reasoning_ans = reasoning_res.metadata.get("answer") or reasoning_res.metadata.get("explanation")
            
            if reasoning_ans:
                final_answer = reasoning_ans
            elif verified_count == 0:
                final_answer = "I found no verified evidence matching your request in the CCTV footage."
            else:
                track_summary = [f"- Track {v['track_id']} ({v['camera_id']}): {v['description'].split('{')[0].strip()}" for v in unique_tracks.values()]
                final_answer = f"I identified {verified_count} individuals in the available CCTV evidence:\n" + "\n".join(track_summary)

        # Determine overall status
        has_error = bool(context.results.get("error"))
        if not has_error and reasoning_res and getattr(reasoning_res, "status", None) == "error":
            has_error = True
            
        error_msg = context.results.get("error")
        if not error_msg and has_error and reasoning_res:
            error_msg = "; ".join(reasoning_res.metadata.get("errors", [])) if hasattr(reasoning_res, "metadata") else "Reasoning failed"
            final_answer = "Reasoning failed: " + error_msg

        return {
            "status": "error" if has_error else "success",
            "error": error_msg,
            "final_answer": final_answer,
            "content": final_answer,
            "evidence": formatted_evidence,
            "citations": [cit.model_dump() for cit in context.citations],
            "overall_confidence": 0.95 if verified_count > 0 else 0.0,
            "agent_decisions": context.agent_decisions,
            "execution": {
                "status": "completed",
                "steps": exec_steps
            }
        }
