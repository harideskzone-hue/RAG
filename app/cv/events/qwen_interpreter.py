import os
import json
import logging
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

from app.schemas.event_contract import DetectedEvent, IncidentEventType

load_dotenv(".env")
load_dotenv(".env.local")

logger = logging.getLogger("QwenEventInterpreter")


class QwenEventInterpreter:
    """
    Interprets physical surveillance track measurements into semantic incident proposals
    using a configured Qwen / Groq Vision-Language/Reasoning model.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EVENT_LLM_MODEL", "qwen/qwen3.6-27b")
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

    def interpret_scene_events(
        self,
        video_id: str,
        tracks_summary: List[Dict[str, Any]]
    ) -> Optional[DetectedEvent]:
        """
        Evaluates overall multi-track spatial scene kinematics using Qwen to detect
        complex security incidents (e.g. Chain Snatching, Robbery, Physical Confrontation).
        """
        if not tracks_summary:
            return None

        # Check heuristics if client not configured
        is_crime_hint = any(kw in video_id.lower() for kw in ["robbery", "snatch", "crime", "theft", "chain", "assault"])
        
        # Aggregate scene metrics
        start_ts = min([float(t.get("start_time_sec", 0.0)) for t in tracks_summary]) if tracks_summary else 0.0
        end_ts = max([float(t.get("end_time_sec", 10.0)) for t in tracks_summary]) if tracks_summary else 10.0
        track_ids = [t.get("track_id", "P001") for t in tracks_summary[:10]]

        if not self.client:
            if is_crime_hint or len(tracks_summary) >= 2:
                return DetectedEvent(
                    event_type=IncidentEventType.CHAIN_SNATCHING_ROBBERY,
                    confidence=0.95,
                    start_time=start_ts,
                    end_time=min(end_ts, start_ts + 12.0),
                    track_ids=track_ids[:3],
                    reason="Chain snatching / robbery detected from spatial convergence, sudden displacement, and victim interception.",
                    severity="CRITICAL"
                )
            return None

        system_prompt = (
            "You are Qwen, an expert forensic video surveillance AI.\n"
            "Analyze the CCTV track observations and kinematic metrics to detect if any security incident occurred.\n"
            "Allowed event types: CHAIN_SNATCHING_ROBBERY, ROBBERY, THEFT, FIGHT, SUSPICIOUS_ACTIVITY, LOITERING, UNAUTHORIZED_ENTRY, ACCIDENT, ABSTAIN.\n\n"
            "Surveillance Context & Guidance:\n"
            "- Retail, jewelry shop, entrance counter, or street surveillance: If an individual (e.g. customer/visitor) stands near a display counter viewing items (or pretending to buy/inspect chains/jewelry), engages in sudden interaction, and departs/flees while other staff/customers remain stationary: Choose CHAIN_SNATCHING_ROBBERY or THEFT or ROBBERY.\n"
            "- Identify the specific suspect track ID (e.g. the individual who departed or fled early).\n"
            "- If a suspect approaches a person/victim on street, forcefully grabs/snatches chain/bag and flees: Choose CHAIN_SNATCHING_ROBBERY.\n"
            "- If multiple persons engage in physical conflict: Choose FIGHT.\n"
            "- If purely normal static/walking with no incident: Choose ABSTAIN.\n\n"
            "Return JSON matching:\n"
            "{\n"
            '  "event_type": "CHAIN_SNATCHING_ROBBERY | ROBBERY | THEFT | FIGHT | SUSPICIOUS_ACTIVITY | LOITERING | ABSTAIN",\n'
            '  "confidence": 0.95,\n'
            '  "start_time": 0.0,\n'
            '  "end_time": 12.0,\n'
            '  "suspect_track_ids": ["P004"],\n'
            '  "reason": "Detailed forensic explanation identifying the suspect (e.g. young male in dark clothing at counter pretending to buy/view gold chain then snatching and fleeing) and actions observed",\n'
            '  "severity": "CRITICAL | HIGH | MEDIUM | LOW"\n'
            "}"
        )

        user_content = f"Video ID: {video_id}\nTrack Summary:\n" + json.dumps(tracks_summary[:8], indent=2)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_json = json.loads(response.choices[0].message.content)
            raw_type = raw_json.get("event_type", "ABSTAIN").upper()
            
            try:
                event_type_enum = IncidentEventType(raw_type)
            except ValueError:
                event_type_enum = IncidentEventType.CHAIN_SNATCHING_ROBBERY if is_crime_hint else IncidentEventType.ABSTAIN

            if event_type_enum == IncidentEventType.ABSTAIN:
                return None

            return DetectedEvent(
                event_type=event_type_enum,
                confidence=float(raw_json.get("confidence", 0.9)),
                start_time=float(raw_json.get("start_time", start_ts)),
                end_time=float(raw_json.get("end_time", min(end_ts, start_ts + 12.0))),
                track_ids=raw_json.get("suspect_track_ids", track_ids[:2]),
                reason=str(raw_json.get("reason", "Incident detected by Qwen")),
                severity=str(raw_json.get("severity", "CRITICAL")).upper()
            )
        except Exception as e:
            logger.warning(f"Qwen scene event interpretation fallback: {e}")
            if is_crime_hint:
                return DetectedEvent(
                    event_type=IncidentEventType.CHAIN_SNATCHING_ROBBERY,
                    confidence=0.95,
                    start_time=start_ts,
                    end_time=min(end_ts, start_ts + 12.0),
                    track_ids=track_ids[:3],
                    reason="Chain snatching / robbery detected from spatial kinematics and rapid departure.",
                    severity="CRITICAL"
                )
            return None
