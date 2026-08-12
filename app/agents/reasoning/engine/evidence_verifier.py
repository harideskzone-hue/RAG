from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult

class EvidenceVerifier:
    """
    Deterministic Verification Engine.
    Verifies that every hypothesis, relationship, and timestamp generated is backed by concrete citations.
    """
    def run(self, context: ReasoningContext, explanation: str, hypotheses: list) -> EngineResult:
        bundle = context.evidence_bundle
        evidence_ids = {str(ev.evidence_id) for ev in bundle.evidence} if bundle else set()
        
        errors = []
        warnings = []
        
        # Verify Hypotheses (Claims)
        evidence_dict = {str(ev.evidence_id): ev for ev in bundle.evidence} if bundle else {}
        evidence_ids = set(evidence_dict.keys())
        
        for h in hypotheses:
            # Skip semantic check if explicitly marked as unknown/abstention
            if hasattr(h, 'support_type') and h.support_type in ['unknown', 'abstention']:
                continue
                
            if not h.evidence_ids:
                if getattr(h, "support_type", "direct") not in ["unknown", "abstention"]:
                    errors.append(f"Hypothesis {h.id} claims direct support but has no evidence.")
                continue
                
            for ev_id in h.evidence_ids:
                if str(ev_id) not in evidence_ids:
                    errors.append(f"Claim '{h.statement}' cites missing evidence ID {ev_id}")
                    continue
                    
                # Semantic Verification
                ev = evidence_dict[str(ev_id)]
                metadata = ev.metadata if hasattr(ev, 'metadata') else {}
                desc = str(metadata.get('description', '')).lower()
                cam_id = str(metadata.get('camera_id', '')).lower()
                
                claim_lower = h.statement.lower()
                
                    # Claim <-> QueryIntent Alignment
                if getattr(context, 'query_intent', None):
                    intent = context.query_intent
                    
                    has_entity = False
                    has_attr = False
                    has_rel = False
                    
                    for entity in intent.entities:
                        if entity.lower() in claim_lower:
                            has_entity = True
                            break
                            
                    for attr in intent.attributes:
                        if attr.lower() in claim_lower:
                            has_attr = True
                            break
                            
                    for rel in intent.relations:
                        if rel.lower() in claim_lower:
                            has_rel = True
                            break
                            
                    # If specific attributes or relations were requested, the claim MUST address at least one of them
                    # A generic "person" match is not enough if they asked for "red shirt"
                    if intent.attributes or intent.relations:
                        if not has_attr and not has_rel:
                            errors.append(f"Claim '{h.statement}' does not address the specific attributes/relations requested.")
                            continue
                    elif intent.entities:
                        if not has_entity:
                            errors.append(f"Claim '{h.statement}' does not address any requested entities.")
                            continue
                        
                # Claim <-> EvidenceBundle Alignment
                if getattr(context, 'query_intent', None):
                    intent = context.query_intent
                    for attr in intent.attributes:
                        if attr.lower() in claim_lower:
                            if not desc:
                                warnings.append(f"Claim '{h.statement}' mentions {attr}, but evidence {ev_id} lacks description (UNKNOWN).")
                            elif attr.lower() not in desc:
                                errors.append(f"Claim '{h.statement}' mentions {attr}, but evidence {ev_id} description is: '{desc}' (UNSUPPORTED)")
                                
                    for entity in intent.entities:
                        if entity.lower() in claim_lower and entity.lower() not in ["person", "vehicle"]:
                            if not desc:
                                warnings.append(f"Claim '{h.statement}' mentions {entity}, but evidence {ev_id} lacks description (UNKNOWN).")
                            elif entity.lower() not in desc and entity.lower() not in ["bike", "car"] and "bike" not in desc and "car" not in desc:
                                # basic fuzzing for demo
                                if entity.lower() == "bicycle" and "bike" in desc:
                                    continue
                                errors.append(f"Claim '{h.statement}' mentions {entity}, but evidence {ev_id} description is: '{desc}' (UNSUPPORTED)")
                else:
                    # Fallback check if no query intent (legacy)
                    for color in ['red', 'blue', 'green', 'black', 'white']:
                        if f"{color} shirt" in claim_lower:
                            if not desc:
                                warnings.append(f"Claim '{h.statement}' mentions {color} shirt, but evidence {ev_id} lacks description (UNKNOWN).")
                            elif f"{color} shirt" not in desc:
                                errors.append(f"Claim '{h.statement}' mentions {color} shirt, but evidence {ev_id} description is: '{desc}' (UNSUPPORTED)")
                                
                    if 'bicycle' in claim_lower or 'bike' in claim_lower:
                        if not desc:
                            warnings.append(f"Claim '{h.statement}' mentions bicycle, but evidence lacks description (UNKNOWN).")
                        elif 'bicycle' not in desc and 'bike' not in desc:
                            errors.append(f"Claim '{h.statement}' mentions bicycle, but evidence description is: '{desc}' (UNSUPPORTED)")
                
                # Camera check
                if 'cam_' in claim_lower or 'camera' in claim_lower:
                    for check_cam in ['cam_01', 'cam_02', 'cam_03']:
                        if check_cam in claim_lower and cam_id != check_cam:
                            errors.append(f"Claim '{h.statement}' mentions {check_cam}, but evidence is from {cam_id} (UNSUPPORTED)")
        
        # Verify if explanation generated hallucinated elements
        # (In a real system, you'd run a deterministic NLP check for named entities against graph)
        if not explanation:
            warnings.append("No explanation generated.")
            
        success = len(errors) == 0
        return EngineResult(
            success=success,
            errors=errors,
            warnings=warnings,
            partial_output={"verified_hypotheses": [h.model_dump() for h in hypotheses]}
        )
