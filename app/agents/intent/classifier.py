from typing import Any

from app.agents.intent.enums import Intent
from app.agents.intent.schemas import IntentResult
from app.domain.models.confidence import ConfidenceScore


class HybridIntentClassifier:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def classify(self, query: str) -> IntentResult:
        # Simple intent classification logic (would be replaced with actual LLM/rules)
        query_lower = query.lower().strip()

        # Default fallback
        intent = Intent.UNKNOWN
        entities = {}
        confidence = 0.0
        requires_clarification = False

        import re
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        # Simple keyword-based intent classification
        if bool(query_words.intersection({"hi", "hello", "hey", "greetings"})):
            intent = Intent.GREETING
            confidence = 1.0
        elif bool(query_words.intersection({"camera", "cameras", "status"})):
            intent = Intent.CAMERA_STATUS
            confidence = 0.9
        elif bool(query_words.intersection({"person", "people", "man", "woman", "anyone", "someone"})):
            intent = Intent.PERSON_SEARCH
            confidence = 0.85
        elif bool(query_words.intersection({"vehicle", "car", "truck"})):
            intent = Intent.VEHICLE_SEARCH
            confidence = 0.8
        elif bool(query_words.intersection({"event", "incident", "happened"})):
            intent = Intent.EVENT_SEARCH
            confidence = 0.75
        elif bool(query_words.intersection({"report", "summary", "analytics"})):
            intent = Intent.REPORT
            confidence = 0.8
        elif bool(query_words.intersection({"graph", "knowledge", "connection"})):
            intent = Intent.KNOWLEDGE_GRAPH_UPDATE
            confidence = 0.7
        else:
            # If we can't determine intent, ask for clarification
            requires_clarification = True
            confidence = 0.2

        from app.schemas.context import QueryIntent
        
        # Simple extraction for demo purposes
        # In a real system, an LLM would populate this structure
        extracted_entities = []
        extracted_attributes = []
        extracted_relations = []
        
        if intent == Intent.PERSON_SEARCH:
            extracted_entities.append("person")
            
            # Simple attribute extraction
            for color in ["red", "blue", "green", "black", "white", "yellow"]:
                if f"{color} shirt" in query_lower:
                    extracted_attributes.append(f"{color} shirt")
                if color in query_lower and f"{color} shirt" not in query_lower:
                    # just color
                    pass
            
            if "bike" in query_lower or "bicycle" in query_lower:
                extracted_entities.append("bicycle")
                extracted_relations.append("person riding bicycle")
                
            if "vehicle" in query_lower or "car" in query_lower:
                extracted_entities.append("vehicle")
                extracted_relations.append("person near vehicle")
                
        query_intent = QueryIntent(
            entities=extracted_entities,
            attributes=extracted_attributes,
            relations=extracted_relations,
            raw_query=query
        )

        # If no LLM configured and fast extractor failed
        return IntentResult(
            success=True,
            intent=intent,
            entities=entities,
            query_intent=query_intent,
            confidence=ConfidenceScore(overall=confidence, factors=[]),
            requires_clarification=requires_clarification
        )