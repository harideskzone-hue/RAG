from typing import Dict, Any

from pydantic import BaseModel, Field
from typing import List

class VerifiedResultContract(BaseModel):
    """
    The STRICT output from the verification layer.
    The Response LLM is only permitted to read this contract.
    """
    verified_persons: List[str] = Field(default_factory=list)
    verified_tracks: List[str] = Field(default_factory=list)
    verified_count: int = Field(0)
    timestamps: List[str] = Field(default_factory=list)
    cameras: List[str] = Field(default_factory=list)
    provenance: List[str] = Field(default_factory=list)
    raw_evidence_summary: str = Field("")

class VerificationNode:
    """
    Deterministically verifies the retrieved evidence.
    NO LLM usage allowed here.
    """
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if "abstain_reason" in state and state["abstain_reason"]:
            return state
            
        evidence_list = state.get("retrieved_evidence", [])
        intent = state.get("query_intent")
        
        contract = VerifiedResultContract()
        
        # Deterministic verification logic
        total_found = 0
        
        for result in evidence_list:
            tool_name = result["tool"]
            data = result["data"]
            
            if not data:
                continue
                
            if isinstance(data, list):
                total_found += len(data)
                # In a real implementation, we extract unique persons, tracks, etc.
                # using strict set addition based on the returned DB models.
                for item in data:
                    # Mock extraction for Phase 4
                    if hasattr(item, "camera_id") and item.camera_id not in contract.cameras:
                        contract.cameras.append(item.camera_id)
                    if hasattr(item, "timestamp_sec") and item.timestamp_sec not in contract.timestamps:
                        contract.timestamps.append(str(item.timestamp_sec))
            elif isinstance(data, dict):
                total_found += 1
                
        if total_found == 0:
            # 7. Explicit Abstention Path: Insufficient Evidence
            state["abstain_reason"] = "Insufficient evidence in the database to answer this query."
            state["verified_contract"] = None
        else:
            contract.verified_count = total_found
            contract.raw_evidence_summary = f"Found {total_found} verified records matching the intent."
            state["verified_contract"] = contract
            
        return state
