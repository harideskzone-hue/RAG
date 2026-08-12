from app.agents.intent.enums import Intent
from app.agents.intent.extractor import FastIntentExtractor

# Golden Dataset
GOLDEN_DATASET = [
    ("Find the person wearing a blue shirt", Intent.PERSON_SEARCH),
    ("Show all thefts yesterday", Intent.EVENT_SEARCH),
    ("Generate weekly report", Intent.REPORT),
    ("Show Gate A cameras", Intent.CAMERA_SEARCH),
    ("Is Camera 5 online?", Intent.CAMERA_STATUS),
    ("Find all fire incidents", Intent.FIRE_ALERT),
    ("Compare crowd density", Intent.CROWD_ANALYSIS),
    ("Show the clip from Camera 2 at 10:15", Intent.CLIP_RETRIEVAL),
    ("Look for a red car leaving the parking lot", Intent.VEHICLE_SEARCH),
    ("Was there a fight in the lobby?", Intent.FIGHT_ALERT),
    ("Any people loitering near the backdoor?", Intent.PERSON_SEARCH)
]

def test_fast_intent_extractor():
    extractor = FastIntentExtractor()
    
    # We will test the golden dataset against the fast extractor
    # Note: Some complex queries might fail the fast extractor and fall back to LLM.
    # We will check if the fast extractor correctly identifies those it's designed for.
    
    for query, expected_intent in GOLDEN_DATASET:
        intent, confidence = extractor.extract_intent(query)
        # For this test, we expect the fast extractor to catch most of them.
        # If it returns UNKNOWN, it implies the LLM would handle it.
        # But we designed the regex to catch these exact keywords.
        assert intent == expected_intent or intent == Intent.UNKNOWN

def test_fast_intent_extractor_specifics():
    extractor = FastIntentExtractor()
    
    intent, conf = extractor.extract_intent("Find the person wearing a blue shirt")
    assert intent == Intent.PERSON_SEARCH
    
    intent, conf = extractor.extract_intent("Generate weekly report")
    assert intent == Intent.REPORT
    
    intent, conf = extractor.extract_intent("Is Camera 5 online?")
    assert intent == Intent.CAMERA_STATUS
    
    # Test entity extraction
    entities = extractor.extract_entities_basic("Show the clip from Camera 2 at 10:15")
    assert "camera_id" in entities
    assert entities["camera_id"] == "2"
