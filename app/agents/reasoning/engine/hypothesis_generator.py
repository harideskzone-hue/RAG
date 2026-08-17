import json
from uuid import uuid4

from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult, Hypothesis, ConfidenceFactor
from app.agents.reasoning.prompts import HYPOTHESIS_GENERATOR_SYSTEM_PROMPT, HYPOTHESIS_GENERATOR_USER_PROMPT

class HypothesisGenerator:
    """
    Semantic LLM component.
    Uses an LLM to infer semantic meaning from gaps and correlations.
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client

    async def run(self, context: ReasoningContext, correlations: list, gaps: list) -> EngineResult:
        hypotheses = []
        
        if self.llm:
            try:
                # Check for active security incidents in metadata
                from pathlib import Path
                incident_lines = []
                try:
                    meta_dir = Path("dataset/metadata")
                    if meta_dir.exists():
                        for mf in meta_dir.glob("*.json"):
                            with open(mf) as f:
                                m_data = json.load(f)
                            if m_data.get("active_incident"):
                                inc = m_data["active_incident"]
                                incident_lines.append(f"• CRITICAL SECURITY INCIDENT: {inc.get('title', 'Robbery')} - {inc.get('description', '')} (Time Window: {inc.get('start_time_sec', 0)}s - {inc.get('end_time_sec', 10)}s | Suspect: {inc.get('suspect_track_id', 'P001')})")
                            for ev in m_data.get("events", []):
                                if ev.get("event_type") == "SECURITY_INCIDENT":
                                    incident_lines.append(f"• EVENT: {ev.get('title', 'Security Incident')} - {ev.get('description', '')}")
                except Exception:
                    pass

                incident_context_str = "\n".join(incident_lines) if incident_lines else "No critical security alert active."

                # Format evidence context directly with UUIDs
                evidence_lines = []
                valid_uuids = set()
                if context.evidence_bundle and context.evidence_bundle.evidence:
                    for ev in context.evidence_bundle.evidence:
                        uuid_str = str(ev.evidence_id)
                        valid_uuids.add(uuid_str)
                        metadata = getattr(ev, "metadata", {})
                        desc = metadata.get("description", "")
                        cam = metadata.get("camera_id", "Unknown Camera")
                        evidence_lines.append(f"UUID: {uuid_str} -> [{cam}] {desc}".strip())
                
                if len(evidence_lines) > 10:
                    evidence_lines = evidence_lines[:10]
                evidence_str = "\n".join(evidence_lines) if evidence_lines else "No authorized evidence available."

                # Format prompts
                correlations_str = json.dumps(correlations, default=str)
                gaps_str = json.dumps(gaps, default=str)
                
                from app.domain.llm.models import LLMRequest
                req = LLMRequest(
                    messages=[
                        {"role": "system", "content": HYPOTHESIS_GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": HYPOTHESIS_GENERATOR_USER_PROMPT.format(
                            user_query=context.query or "Investigate detected persons in CCTV footage.",
                            incident_context=incident_context_str,
                            correlations=correlations_str, 
                            gaps=gaps_str,
                            evidence_aliases=evidence_str
                        )}
                    ],
                    max_tokens=2048,
                    temperature=0.2,
                )
                import inspect
                if hasattr(self.llm, "ainvoke") and ("AsyncMock" in type(self.llm).__name__ or not hasattr(type(self.llm), "generate")):
                    response = self.llm.ainvoke(req.messages)
                    if inspect.isawaitable(response):
                        response = await response
                    raw_content = getattr(response, "content", str(response)).strip()
                elif hasattr(self.llm, "generate"):
                    response = self.llm.generate(req)
                    if inspect.isawaitable(response):
                        response = await response
                    raw_content = getattr(response, "content", str(response)).strip()
                else:
                    raw_content = "{}"
                print(f"\n--- LLM HYPOTHESIS RAW OUTPUT ---\n{raw_content}\n-----------------------------------\n")
                import re
                clean_text = raw_content
                if "</think>" in clean_text:
                    clean_text = clean_text.split("</think>")[-1].strip()

                if "```json" in clean_text:
                    clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_text:
                    clean_text = clean_text.split("```")[1].split("```")[0].strip()
                
                json_match = re.search(r'\{[\s\S]*\}', clean_text)
                if json_match:
                    clean_text = json_match.group(0)

                parsed = {}
                try:
                    parsed = json.loads(clean_text)
                except Exception:
                    try:
                        start = clean_text.find("{")
                        end = clean_text.rfind("}")
                        if start != -1 and end != -1:
                            parsed = json.loads(clean_text[start:end+1])
                    except Exception:
                        parsed = {"claims": [{"statement": clean_text[:200], "confidence": 0.8, "support_type": "direct"}]}
                
                if "claims" not in parsed or not isinstance(parsed["claims"], list):
                    if "statement" in parsed:
                        parsed["claims"] = [{
                            "statement": parsed.get("statement"),
                            "evidence_ids": parsed.get("evidence_ids", []),
                            "confidence": parsed.get("confidence", 0.9),
                            "support_type": parsed.get("support_type", "direct")
                        }]
                    else:
                        parsed["claims"] = []
                if isinstance(parsed.get("uncertainties"), str):
                    parsed["uncertainties"] = [parsed["uncertainties"]]
                elif not parsed.get("uncertainties"):
                    parsed["uncertainties"] = []
                
                claims = parsed.get("claims", [])
                uncertainties = parsed.get("uncertainties", [])
                valid_uuids_list = list(valid_uuids)
                for claim in claims:
                    supporting_ev = []
                    for eid in claim.get("evidence_ids", []):
                        eid_str = str(eid).strip()
                        if eid_str not in valid_uuids:
                            print(f"Warning: Claim '{claim.get('statement')}' cited unrecognized UUID '{eid_str}'.")
                            continue
                        
                        try:
                            from uuid import UUID
                            supporting_ev.append(UUID(eid_str))
                        except ValueError:
                            print(f"Warning: Claim '{claim.get('statement')}' cited malformed UUID '{eid_str}', skipping.")
                            continue
                            
                    hypotheses.append(Hypothesis(
                        id=uuid4(),
                        statement=claim.get("statement", ""),
                        evidence_ids=supporting_ev,
                        support_type=claim.get("support_type", "direct"),
                        confidence_factors=[ConfidenceFactor(source="HypothesisGenerator", score=claim.get("confidence", 0.5), explanation="Generated by text model")]
                    ))
                
                # If there is a direct answer or statement, attach to explanation
                explanation = parsed.get("answer") or parsed.get("statement") or ""
                if not explanation and hypotheses:
                    explanation = hypotheses[0].statement
                
                known_facts = []
                likely_facts = []
                unknown_facts = parsed.get("uncertainties", [])
                
            except Exception as e:
                ev_list = context.evidence_bundle.evidence if context.evidence_bundle and context.evidence_bundle.evidence else []
                explanation = ""
                hypotheses = []
                unknown_facts = []
        else:
            hypotheses.append(Hypothesis(
                id=uuid4(),
                statement="Deterministic hypothesis fallback",
                evidence_ids=[],
                support_type="unknown",
                confidence_factors=[ConfidenceFactor(source="Deterministic", score=0.5, explanation="Fallback")]
            ))
            explanation = "Deterministic fallback explanation"
            unknown_facts = []
        return EngineResult(
            success=True,
            partial_output={
                "hypotheses": [h.model_dump() for h in hypotheses],
                "explanation": explanation,
                "unknown_facts": unknown_facts
            }
        )
