import json
import logging
import re
from typing import Any

from app.agents.intent.enums import Intent
from app.agents.intent.schemas import IntentResult
from app.agents.intent.intent_prompts import INTENT_SYSTEM_PROMPT, INTENT_USER_PROMPT
from app.domain.models.confidence import ConfidenceScore
from app.schemas.context import QueryIntent

logger = logging.getLogger(__name__)


class HybridIntentClassifier:
    """
    Semantic-first intent classifier.
    All query interpretation, entity/attribute extraction, relationship understanding,
    and search operations are dynamically model-driven.
    Deterministic logic is restricted to greeting control flow and LLM failure fallback.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def classify(self, query: str) -> IntentResult:
        query_clean = query.strip()
        query_lower = query_clean.lower()

        # Deterministic control flow for simple greetings
        words = set(re.findall(r'\b\w+\b', query_lower))
        if bool(words.intersection({"hi", "hello", "hey", "greetings"})) and len(words) <= 3:
            query_intent = QueryIntent(domain="general", operation="greeting", raw_query=query_clean)
            return IntentResult(
                success=True,
                intent=Intent.GREETING,
                domain="general",
                operation="greeting",
                entities={"description": query_clean},
                query_intent=query_intent,
                confidence=ConfidenceScore(overall=1.0, factors=[]),
                requires_clarification=False
            )

        # Deterministic control flow for activity / behavior questions
        if re.search(r'\b(what are they doing|what are people doing|what is happening|what is going on|what activities|activity|actions|doing in the video|doing in the cctv)\b', query_lower):
            query_intent = QueryIntent(
                domain="investigation",
                operation="behavioral_investigation",
                search_operations=["metadata_query", "event_query", "vector_person"],
                required_capabilities=["person_search", "behavior_analysis", "video_analysis"],
                raw_query=query_clean
            )
            return IntentResult(
                success=True,
                intent=Intent.BEHAVIORAL_INVESTIGATION,
                domain="investigation",
                operation="behavioral_investigation",
                entities={"description": query_clean, "activity": "actions in scene"},
                query_intent=query_intent,
                confidence=ConfidenceScore(overall=1.0, factors=[]),
                requires_clarification=False
            )

        # Deterministic control flow for capability questions
        if re.search(r'\b(can do|capabilities|capability|what can vista|how does vista work|what are you|who are you|vista features|help)\b', query_lower):
            query_intent = QueryIntent(domain="general", operation="capability_explanation", search_operations=[], raw_query=query_clean)
            return IntentResult(
                success=True,
                intent=Intent.CAPABILITY_EXPLANATION,
                domain="general",
                operation="capability_explanation",
                entities={"description": query_clean},
                query_intent=query_intent,
                confidence=ConfidenceScore(overall=1.0, factors=[]),
                requires_clarification=False
            )

        # Deterministic control flow for time questions
        if re.search(r'\b(what is the time|current time|what time is it|time right now|what date|today\'s date)\b', query_lower):
            query_intent = QueryIntent(domain="general", operation="time_query", search_operations=["time_query"], raw_query=query_clean)
            return IntentResult(
                success=True,
                intent=Intent.TIME_QUERY,
                domain="general",
                operation="time_query",
                entities={"description": query_clean},
                query_intent=query_intent,
                confidence=ConfidenceScore(overall=1.0, factors=[]),
                requires_clarification=False
            )

        # Dynamic LLM-driven classification takes priority when LLM client is configured
        if self.llm_client:
            try:
                response = await self.llm_client.ainvoke([
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": INTENT_USER_PROMPT.format(query=query_clean)}
                ])
                raw_content = response.content.strip()
                
                # Extract JSON object from LLM response
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if json_match:
                    raw_content = json_match.group(0)

                parsed = json.loads(raw_content)
                
                # Parse domain and operation
                domain_str = str(parsed.get("domain", "investigation")).lower()
                operation_str = str(parsed.get("operation", "")).lower()
                
                # Parse intent enum safely
                intent_str = str(parsed.get("intent", "unknown")).lower()
                try:
                    intent = Intent(intent_str)
                except ValueError:
                    if domain_str == "general":
                        intent = Intent.CAPABILITY_EXPLANATION if "capab" in operation_str else Intent.GENERAL_QUERY
                    else:
                        intent = Intent.UNKNOWN

                entities_list = parsed.get("entities") or []
                if not isinstance(entities_list, list): entities_list = [str(entities_list)]
                
                attributes_list = parsed.get("attributes") or []
                if not isinstance(attributes_list, list): attributes_list = [str(attributes_list)]
                
                relations_list = parsed.get("relations") or []
                if not isinstance(relations_list, list): relations_list = [str(relations_list)]
                
                temporal_list = parsed.get("temporal_constraints") or []
                if not isinstance(temporal_list, list): temporal_list = [str(temporal_list)]
                
                spatial_list = parsed.get("spatial_constraints") or []
                if not isinstance(spatial_list, list): spatial_list = [str(spatial_list)]
                
                search_ops_list = parsed.get("search_operations") or []
                if not isinstance(search_ops_list, list): search_ops_list = [str(search_ops_list)]
                
                req_caps_list = parsed.get("required_capabilities") or []
                if not isinstance(req_caps_list, list): req_caps_list = [str(req_caps_list)]

                conf_val = float(parsed.get("confidence") or 0.85)

                target_type_str = str(parsed.get("target_type", "")).lower()
                semantic_constraints_list = parsed.get("semantic_constraints") or []
                if not isinstance(semantic_constraints_list, list): semantic_constraints_list = [str(semantic_constraints_list)]

                # FOR GENERAL DOMAIN QUERIES, ENSURE search_operations is EMPTY
                if domain_str == "general" or operation_str in ["capability_explanation", "greeting"] or intent in [Intent.CAPABILITY_EXPLANATION, Intent.GREETING, Intent.GENERAL_QUERY]:
                    search_ops_list = []
                    req_caps_list = []
                elif not search_ops_list and domain_str == "investigation":
                    search_ops_list.append("vector_person")

                query_intent = QueryIntent(
                    domain=domain_str,
                    operation=operation_str,
                    target_type=target_type_str,
                    semantic_constraints=semantic_constraints_list,
                    entities=entities_list,
                    attributes=attributes_list,
                    relations=relations_list,
                    temporal_constraints=temporal_list,
                    spatial_constraints=spatial_list,
                    search_operations=search_ops_list,
                    required_capabilities=req_caps_list,
                    raw_query=query_clean
                )

                entities_dict = {
                    "description": query_clean,
                    "attributes": attributes_list,
                    "entities": entities_list,
                    "relations": relations_list
                }

                return IntentResult(
                    success=True,
                    intent=intent,
                    domain=domain_str,
                    operation=operation_str,
                    entities=entities_dict,
                    query_intent=query_intent,
                    confidence=ConfidenceScore(overall=conf_val, factors=[]),
                    requires_clarification=parsed.get("requires_clarification", False)
                )

            except Exception as e:
                logger.warning(f"LLM intent classification failed: {e}. Falling back to default RAG plan.")

        # Deterministic control flow for counting queries
        if any(phrase in query_lower for phrase in ["how many", "count of", "number of", "how many person", "how many people"]):
            sem_c = []
            if bool(re.search(r'\b(male|man|men)\b', query_lower)) and not bool(re.search(r'\b(woman|women|female)\b', query_lower)):
                sem_c.append("gender=male")
            elif bool(re.search(r'\b(woman|women|female|lady|ladies)\b', query_lower)):
                sem_c.append("gender=female")
            elif bool(re.search(r'\b(kid|kids|child|children|boy|boys|girl|girls)\b', query_lower)):
                sem_c.append("age_group=child")

            search_ops = ["vector_vehicle"] if any(w in query_lower for w in ["vehicle", "car", "bike", "truck"]) else ["vector_person"]
            query_intent = QueryIntent(
                domain="investigation",
                operation="count",
                target_type="person",
                semantic_constraints=sem_c,
                attributes=sem_c,
                search_operations=search_ops,
                required_capabilities=["person_search", "identity_aggregation"],
                raw_query=query_clean
            )
            return IntentResult(
                success=True,
                intent=Intent.COUNT,
                domain="investigation",
                operation="count",
                entities={"description": query_clean},
                query_intent=query_intent,
                confidence=ConfidenceScore(overall=1.0, factors=[]),
                requires_clarification=False
            )

        # Deterministic control flow for event search / fire queries
        if any(phrase in query_lower for phrase in ["fire", "fire accident", "explosion", "accident", "incident"]):
            sem_c = ["event_type=fire"] if "fire" in query_lower else []
            query_intent = QueryIntent(
                domain="investigation",
                operation="event_search",
                target_type="event",
                semantic_constraints=sem_c,
                search_operations=["event_query"],
                required_capabilities=["event_search", "video_analysis"],
                raw_query=query_clean
            )
            return IntentResult(
                success=True,
                intent=Intent.EVENT_SEARCH,
                domain="investigation",
                operation="event_search",
                entities={"description": query_clean},
                query_intent=query_intent,
                confidence=ConfidenceScore(overall=1.0, factors=[]),
                requires_clarification=False
            )

        # Deterministic control flow for behavioral investigation queries
        if any(phrase in query_lower for phrase in ["suspicious", "suspect", "suspects", "unusual behavior", "acting suspicious"]):
            query_intent = QueryIntent(
                domain="investigation",
                operation="behavioral_investigation",
                target_type="person",
                search_operations=["vector_person", "event_query"],
                required_capabilities=["person_search", "video_analysis", "behavior_analysis"],
                raw_query=query_clean
            )
            return IntentResult(
                success=True,
                intent=Intent.BEHAVIORAL_INVESTIGATION,
                domain="investigation",
                operation="behavioral_investigation",
                entities={"description": query_clean},
                query_intent=query_intent,
                confidence=ConfidenceScore(overall=1.0, factors=[]),
                requires_clarification=False
            )
        # Safe fallback when LLM fails or is unconfigured
        cctv_keywords = {"cctv", "camera", "men", "women", "man", "woman", "person", "vehicle", "footage", "frame"}
        is_investigation = bool(words.intersection(cctv_keywords))
        domain_fallback = "investigation" if is_investigation else "general"
        intent_fallback = Intent.PERSON_SEARCH if is_investigation else Intent.UNKNOWN
        default_search_ops = ["vector_person"] if is_investigation else []

        query_intent = QueryIntent(
            domain=domain_fallback,
            operation="person_search" if is_investigation else "",
            entities=[],
            attributes=[],
            relations=[],
            search_operations=default_search_ops,
            raw_query=query_clean
        )
        return IntentResult(
            success=True,
            intent=intent_fallback,
            domain=domain_fallback,
            operation="person_search" if is_investigation else "",
            entities={"description": query_clean},
            query_intent=query_intent,
            confidence=ConfidenceScore(overall=0.3, factors=[]),
            requires_clarification=False
        )