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
    using a configured Groq LLM (defaults to qwen/qwen3.6-27b).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EVENT_LLM_MODEL", "qwen/qwen3.6-27b")
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

    def interpret_track_events(self, track_summary: Dict[str, Any]) -> DetectedEvent:
        """
        Sends deterministic physical metrics to Qwen and parses the structured response.
        """
        if not self.client:
            logger.warning("Groq client not configured, returning ABSTAIN.")
            return DetectedEvent(
                event_type=IncidentEventType.ABSTAIN,
                confidence=0.0,
                start_time=track_summary.get("start_time", 0.0),
                end_time=track_summary.get("end_time", 0.0),
                track_ids=[track_summary.get("track_id", "")],
                reason="LLM client not configured",
                severity="LOW"
            )

        system_prompt = (
            "You are an expert AI video surveillance incident interpreter.\n"
            "Given physical measurements of a tracked entity in CCTV footage, analyze if the behavior represents a safety incident:\n"
            "Options: LOITERING, UNAUTHORIZED_ENTRY, FIGHT, ACCIDENT, SUSPICIOUS_ACTIVITY, CROWD, FIRE, ABSTAIN.\n\n"
            "Guidance:\n"
            "- Normal walking / transit through frame: Choose ABSTAIN.\n"
            "- Lingering in an area with low displacement over extended duration: Choose LOITERING or SUSPICIOUS_ACTIVITY.\n"
            "- Sudden high-velocity changes or rapid vertical collapse: Choose FIGHT or ACCIDENT.\n"
            "- If ambiguous, uncertain, or normal activity: Choose ABSTAIN.\n\n"
            "You MUST respond with a valid JSON object matching this schema:\n"
            "{\n"
            '  "event_type": "LOITERING | UNAUTHORIZED_ENTRY | FIGHT | ACCIDENT | SUSPICIOUS_ACTIVITY | CROWD | FIRE | ABSTAIN",\n'
            '  "confidence": 0.85,\n'
            '  "start_time": 10.0,\n'
            '  "end_time": 55.0,\n'
            '  "track_ids": ["P014"],\n'
            '  "reason": "Detailed factual reasoning based on the provided metrics",\n'
            '  "severity": "LOW | MEDIUM | HIGH | CRITICAL"\n'
            "}"
        )

        user_content = (
            f"Physical Track Measurements:\n"
            f"- Track ID: {track_summary.get('track_id')}\n"
            f"- Camera ID: {track_summary.get('camera_id')}\n"
            f"- Start Time: {track_summary.get('start_time')}s, End Time: {track_summary.get('end_time')}s\n"
            f"- Duration: {track_summary.get('duration_sec')}s, Frame Count: {track_summary.get('frame_count')}\n"
            f"- Total Path Length: {track_summary.get('total_path_length_px')} px\n"
            f"- Net Displacement: {track_summary.get('net_displacement_px')} px\n"
            f"- Max Dispersion Radius: {track_summary.get('dispersion_radius_px')} px\n"
            f"- Average Speed: {track_summary.get('avg_speed_px_per_sec')} px/s\n"
            f"- Max Speed Spike: {track_summary.get('max_speed_px_per_sec')} px/s\n"
            f"- Average Height: {track_summary.get('avg_height_px')} px, Min Height: {track_summary.get('min_height_px')} px\n"
        )

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
            
            raw_event_type = raw_json.get("event_type", "ABSTAIN").upper()
            try:
                event_type_enum = IncidentEventType(raw_event_type)
            except ValueError:
                event_type_enum = IncidentEventType.ABSTAIN

            return DetectedEvent(
                event_type=event_type_enum,
                confidence=float(raw_json.get("confidence", 0.5)),
                start_time=float(raw_json.get("start_time", track_summary.get("start_time", 0.0))),
                end_time=float(raw_json.get("end_time", track_summary.get("end_time", 0.0))),
                track_ids=[track_summary.get("track_id", "")],
                reason=str(raw_json.get("reason", "Analyzed by Qwen interpreter")),
                severity=str(raw_json.get("severity", "MEDIUM")).upper()
            )
        except Exception as e:
            logger.error(f"Error calling Qwen for event interpretation: {e}")
            return DetectedEvent(
                event_type=IncidentEventType.ABSTAIN,
                confidence=0.0,
                start_time=track_summary.get("start_time", 0.0),
                end_time=track_summary.get("end_time", 0.0),
                track_ids=[track_summary.get("track_id", "")],
                reason=f"Interpretation fallback: {e}",
                severity="LOW"
            )
