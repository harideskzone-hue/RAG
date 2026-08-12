from enum import Enum


class Intent(str, Enum):
    PERSON_SEARCH = "person_search"
    VEHICLE_SEARCH = "vehicle_search"
    EVENT_SEARCH = "event_search"
    REPORT = "report"
    CAMERA_STATUS = "camera_status"
    CAMERA_SEARCH = "camera_search"
    TIMELINE_SEARCH = "timeline_search"
    CLIP_RETRIEVAL = "clip_retrieval"
    CROWD_ANALYSIS = "crowd_analysis"
    FIRE_ALERT = "fire_alert"
    FIGHT_ALERT = "fight_alert"
    LOITERING = "loitering"
    GREETING = "greeting"
    UNKNOWN = "unknown"
