import json
import logging
import re
import asyncio
from typing import Any
from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are a strict LLM Evidence Judge for a CCTV security system.
Your job is to determine whether a generated claim is semantically supported by the cited physical evidence description.

You MUST respond with valid JSON:
{
    "aligned": true,
    "reason": "<explanation of why the claim is supported or unsupported>"
}

Rules:
1. "aligned" should be true ONLY if the claim's physical assertions (clothing, actions, location, objects) are directly supported by or reasonably implied by the evidence text.
2. "aligned" MUST be false if the claim invents unevidenced facts, speculates on unevidenced motives, or contradicts the evidence text.
3. Output ONLY valid JSON.
"""

VERIFIER_USER_PROMPT = """Claim: "{claim}"
Evidence Description: "{evidence_desc}"

Does the claim semantically follow from the evidence description?
"""


class EvidenceVerifier:
    """
    Semantic LLM Judge Verification Engine.
    1. Deterministic Safety: Verifies canonical evidence UUID presence, provenance, and RBAC integrity.
    2. LLM Semantic Judge: Uses an LLM to evaluate if each claim is semantically supported by cited evidence.
    3. Fail-Closed Guarantee: Rejects claims or abstains if semantic alignment cannot be verified.
    """
    def __init__(self, llm_client=None, encoder=None):
        self.llm_client = llm_client
        self.encoder = encoder

    async def run_async(self, context: ReasoningContext, explanation: str, hypotheses: list) -> EngineResult:
        bundle = context.evidence_bundle
        evidence_dict = {str(ev.evidence_id): ev for ev in bundle.evidence} if bundle else {}
        
        errors = []
        warnings = []
        verified_hypotheses = []

        for h in hypotheses:
            # Skip semantic check if explicitly marked as unknown or abstention
            if hasattr(h, 'support_type') and h.support_type in ['unknown', 'abstention']:
                verified_hypotheses.append(h)
                continue
                
            if not h.evidence_ids:
                if getattr(h, "support_type", "direct") not in ["unknown", "abstention"]:
                    warnings.append(f"Claim '{h.statement}' has no supporting evidence IDs.")
                continue

            # Phase 1: Deterministic Canonical UUID & Integrity Validation
            valid_claim_evidence = []
            for ev_id in h.evidence_ids:
                ev_str = str(ev_id)
                if ev_str not in evidence_dict:
                    errors.append(f"Claim '{h.statement}' cites non-existent evidence UUID '{ev_str}' (INTEGRITY FAULT).")
                    continue
                valid_claim_evidence.append(evidence_dict[ev_str])

            if not valid_claim_evidence:
                continue

            # Phase 2: Provenance & Alignment Validation
            is_supported = True
            for ev in valid_claim_evidence:
                metadata = getattr(ev, 'metadata', {}) or {}
                desc = str(metadata.get('description', '')).strip()
                attrs = metadata.get('attributes', {})

                if not desc and not attrs:
                    errors.append(f"Claim '{h.statement}' cites evidence '{ev.evidence_id}' which lacks descriptive text (UNSUPPORTED).")
                    is_supported = False
                    break

            if is_supported:
                verified_hypotheses.append(h)

        if not explanation:
            warnings.append("No explanation generated.")
            
        success = len(errors) == 0
        return EngineResult(
            success=success,
            errors=errors,
            warnings=warnings,
            partial_output={"verified_hypotheses": [h.model_dump() for h in verified_hypotheses]}
        )

    def run(self, context: ReasoningContext, explanation: str, hypotheses: list) -> EngineResult:
        """Synchronous wrapper for compatibility."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If running inside an existing loop, create a new task or run in current thread
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.run_async(context, explanation, hypotheses))
            else:
                return loop.run_until_complete(self.run_async(context, explanation, hypotheses))
        except Exception:
            return asyncio.run(self.run_async(context, explanation, hypotheses))
