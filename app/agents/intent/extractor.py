import re
from typing import Any

from app.agents.intent.enums import Intent


class FastIntentExtractor:
    """
    Fast Rule Layer to classify intents using regex and keywords.
    Reduces latency and cost for common deterministic queries.
    """
    
    def __init__(self):
        self.rules = [
            (re.compile(r'^(hi|hello|hey|greetings|who are you)(?:\s+.*)?$', re.IGNORECASE), Intent.GREETING),
            (re.compile(r'\b(find|show|search)\b.*\b(person|guy|man|woman|wearing)\b', re.IGNORECASE), Intent.PERSON_SEARCH),
            (re.compile(r'\b(any|about|tell me|find|show|where is)\b.*\b(suspects?|suspicious|person|people|guy|man|woman|wearing)\b', re.IGNORECASE), Intent.PERSON_SEARCH),
            (re.compile(r'\b(suspects?|suspicious (person|people|activity|behavior))\b', re.IGNORECASE), Intent.PERSON_SEARCH),
            (re.compile(r'\b(person|guy|man|woman|anyone)\b.*\b(in|wearing)\b.*\b(shirt|pants|jacket|hat)\b', re.IGNORECASE), Intent.PERSON_SEARCH),
            (re.compile(r'\b(blue|red|green|black|white|yellow)\b.*\b(shirt|pants|jacket|hat)\b.*\b(person|guy|man|woman)\b', re.IGNORECASE), Intent.PERSON_SEARCH),
            (re.compile(r'\b(find|show|search)\b.*\b(car|vehicle|truck|bike)\b', re.IGNORECASE), Intent.VEHICLE_SEARCH),
            (re.compile(r'\b(weekly|monthly|generate|create)\b.*\b(report|stats)\b', re.IGNORECASE), Intent.REPORT),
            (re.compile(r'\b(clip|video|recording)\b.*\b(camera|cam)\b', re.IGNORECASE), Intent.CLIP_RETRIEVAL),
            (re.compile(r'\b(is|are)\b.*\b(camera|cam)\b.*\b(online|offline|working)\b', re.IGNORECASE), Intent.CAMERA_STATUS),
            (re.compile(r'\b(show|list)\b.*\b(cameras?|cams?)\b', re.IGNORECASE), Intent.CAMERA_SEARCH),
            (re.compile(r'\b(fire|smoke)\b', re.IGNORECASE), Intent.FIRE_ALERT),
            (re.compile(r'\b(fight|violence|punch)\b', re.IGNORECASE), Intent.FIGHT_ALERT),
            (re.compile(r'\b(crowd|density|how many people)\b', re.IGNORECASE), Intent.CROWD_ANALYSIS),
            (re.compile(r'\b(thefts?|incidents?|accidents?)\b', re.IGNORECASE), Intent.EVENT_SEARCH),
        ]

    def extract_intent(self, query: str) -> tuple[Intent, float]:
        """
        Returns (Intent, Confidence)
        """
        for pattern, intent in self.rules:
            if pattern.search(query):
                # Deterministic high confidence for simple rules
                return intent, 0.95
                
        return Intent.UNKNOWN, 0.0

    def extract_entities_basic(self, query: str) -> dict[str, Any]:
        """
        Basic regex extraction for entities if needed.
        (LLM will handle complex entities).
        """
        entities = {}
        # Simple camera ID extraction
        cam_match = re.search(r'(camera|cam)\s*(\w+)', query, re.IGNORECASE)
        if cam_match:
            entities['camera_id'] = cam_match.group(2)
        return entities
