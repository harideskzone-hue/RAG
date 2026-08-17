import re
from typing import Dict, Any

class GroundingValidatorNode:
    """
    Deterministically validates the LLM's final response against the VerifiedResultContract.
    If the LLM introduces numbers or specific IDs that contradict the verified evidence,
    it rejects the response and triggers an ABSTAIN/Regeneration.
    """
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = state.get("final_response", "")
        contract = state.get("verified_contract")
        abstain_reason = state.get("abstain_reason")
        
        # If we already abstained, no need to ground the generated abstain message
        # assuming the Response LLM just produced a safe apology.
        if abstain_reason or not contract:
            return state
            
        # Deterministic Grounding Checks
        is_valid = True
        violation_reason = ""
        
        # 1. Check if LLM hallucinates numbers not matching verified count
        # A simple check: extract all integers from response and ensure they 
        # do not contradict the primary verified counts.
        numbers_in_response = set(re.findall(r'\b\d+\b', response))
        # Filter out common linguistic numbers or non-evidence numbers if needed, 
        # but for strict compliance, we ensure the verified_count is present if any count is stated.
        # Actually, a better check: if the LLM states a count, it MUST match verified_count.
        # If the user asked "how many" and the LLM says "999", 999 must be in the contract.
        
        # If verified_count > 0, we expect that number to be in the response or at least not contradicted.
        # This is a basic implementation of a deterministic grounding check.
        if contract.verified_count > 0:
            if str(contract.verified_count) not in numbers_in_response:
                # If they used numbers but not the correct count, they might have hallucinated
                # e.g., "I found 999 people"
                if len(numbers_in_response) > 0:
                    is_valid = False
                    violation_reason = f"Response contains unauthorized numbers {numbers_in_response}. Expected count: {contract.verified_count}"
        
        # 2. Check for Hallucinated IDs (e.g., PXXXX)
        ids_in_response = set(re.findall(r'\bP\d{4,}\b', response))
        for person_id in ids_in_response:
            if person_id not in contract.verified_persons:
                is_valid = False
                violation_reason = f"Response introduces unverified Person ID: {person_id}"
                break
                
        # 3. Check for Hallucinated Cameras (e.g., CAM_XX)
        cams_in_response = set(re.findall(r'\bCAM_\w+\b', response, re.IGNORECASE))
        valid_cams = set([c.upper() for c in getattr(contract, "cameras", [])])
        for cam_id in cams_in_response:
            if valid_cams and cam_id.upper() not in valid_cams:
                is_valid = False
                violation_reason = f"Response introduces unverified Camera ID: {cam_id}"
                break

        # 4. Check for Hallucinated Event IDs (e.g., EVT_XXXXXX)
        events_in_response = set(re.findall(r'\bEVT_\w+\b', response, re.IGNORECASE))
        verified_event_ids = set()
        if hasattr(contract, "verified_events"):
            verified_event_ids = set([e.event_id.upper() for e in contract.verified_events])
        for evt_id in events_in_response:
            if verified_event_ids and evt_id.upper() not in verified_event_ids:
                is_valid = False
                violation_reason = f"Response introduces unverified Event ID: {evt_id}"
                break

        # 5. Check for Hallucinated Media Clip URLs
        urls_in_response = set(re.findall(r'/media/events/\S+\.mp4', response))
        verified_clip_urls = set()
        if hasattr(contract, "verified_events"):
            verified_clip_urls = set([e.clip_url for e in contract.verified_events if e.clip_url])
        for url in urls_in_response:
            if verified_clip_urls and url not in verified_clip_urls:
                is_valid = False
                violation_reason = f"Response introduces unverified Evidence Clip URL: {url}"
                break
                
        if not is_valid:
            # Trigger Regeneration or ABSTAIN
            state["abstain_reason"] = f"Deterministic Grounding Validator rejected LLM response: {violation_reason}"
            state["final_response"] = "The system cannot safely answer this query as it may contain hallucinated evidence."
            state["grounding_valid"] = False
        else:
            state["grounding_valid"] = True
            
        return state
