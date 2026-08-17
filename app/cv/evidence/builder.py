import uuid
from typing import List, Dict, Any
from app.schemas.evidence_contract import EvidenceContract, EvidenceProvenance, EvidenceSubject

class EvidenceBuilder:
    """Builds EvidenceContracts from CV track observations."""
    
    @staticmethod
    def build_from_observations(observations: List[Dict[str, Any]]) -> List[EvidenceContract]:
        """
        Converts aggregator observations into strict EvidenceContract objects.
        """
        contracts = []
        for obs in observations:
            # Reconstruct evidence UUID if possible, or generate a new one
            ev_id_str = obs.get("evidence_id")
            try:
                # evidence_id might be "obs_XXXX", we need a valid UUID for the schema
                # We'll just generate a valid UUID and keep the 'obs' id in extra for reference
                # if it's not a real UUID format.
                ev_uuid = uuid.UUID(ev_id_str)
            except (ValueError, TypeError, AttributeError):
                ev_uuid = uuid.uuid4()

            provenance = EvidenceProvenance(
                video_id=obs.get("video_id"),
                camera_id=obs.get("camera_id"),
                track_id=obs.get("track_id"),
                source_type="video_analysis",
                video_timestamp_sec=obs.get("timestamp_sec"),
                frame_number=obs.get("frame_index")
            )
            
            subject = EvidenceSubject(
                entity_type="person",
                track_id=obs.get("track_id"),
                description=""
            )
            
            contract = EvidenceContract(
                evidence_id=ev_uuid,
                provenance=provenance,
                subject=subject,
                confidence=obs.get("confidence", 0.0),
                source="CV Pipeline",
                observation={
                    "entity_type": "person",
                    "track_id": obs.get("track_id"),
                    "frame_number": obs.get("frame_index"),
                    "timestamp_sec": obs.get("timestamp_sec"),
                    "bbox": obs.get("bbox"),
                    "confidence": obs.get("confidence", 0.0),
                    "original_evidence_id": ev_id_str
                }
            )
            # Remove attributes entirely for Phase 1 to avoid "gender": null semantics
            contract.attributes.model_fields_set.clear()
            contracts.append(contract)
            
        return contracts
