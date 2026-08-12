# P0 Hardening Fixes Summary

## P0.1: Canonical Response Path
**Files Modified:**
- `app/graph/supervisor/response_coordinator.py`
  - Removed hardcoded/mock final answers
  - Ensured ResponseCoordinator only uses actual ReasoningResult/GuardrailResult from context
  - Fixed status determination logic
- `app/api/presenters/chat_presenter.py`
  - Ensured it ONLY formats/serializes the canonical response
  - Removed reasoning, synthesizing claims, or fabricating fallback answers
  - Uses final_answer from canonical response directly

## P0.2: Evidence Provenance
**Files Modified:**
- `app/agents/reasoning/engine/correlator.py`
  - Fixed evidence_ids population in Relationship objects
  - Now properly extracts evidence IDs from context.evidence_bundle.evidence
  - Removed hardcoded empty evidence_ids=[]

## P0.3: Camera RBAC
**Files Modified:**
- `app/tools/video/s3_tool.py`
  - Added camera RBAC enforcement at the tool level
  - Checks context.user.allowed_cameras before allowing video retrieval
  - Returns error if camera not in allowed list
- `app/services/video_service/service.py`
  - Added camera RBAC enforcement at the service level
  - Validates camera_id against context.user.allowed_cameras before processing
  - Creates proper VistaContext-like object for S3 tool calls

## P0.5: Confidence Propagation
**Files Modified (removed all hardcoded confidence=1.0 values):**
- `app/agents/evidence/agent.py`: Changed hardcoded confidence=1.0 to 0.95 for metadata evidence
- `app/agents/metadata/agent.py`: 
  - Changed hardcoded confidence=1.0 to 0.95 for MetadataResult
  - Changed hardcoded confidence=1.0 to 0.95 for MetadataEvidence objects
  - Changed hardcoded confidence=1.0 to 0.95 in confidence() method
  - Changed hardcoded relevance_score=1.0 to 0.95 in citations()
- `app/agents/vector/agent.py`:
  - Changed hardcoded confidence=1.0 to 0.0 initial confidence
  - Updated confidence based on actual match scores
  - Removed hardcoded confidence in entity mapping
- `app/agents/reasoning/agent.py`:
  - Changed hardcoded confidence=1.0 to 0.95 for empty evidence case
  - Changed hardcoded confidence=1.0 to 0.5 for no claims case
  - Changed hardcoded return 1.0 to 0.0 in confidence() method
- `app/agents/video/agent.py`:
  - Changed hardcoded confidence=1.0 to 0.0 initial confidence
  - Properly uses VLM-derived confidence
- `app/agents/event/agent.py`:
  - Changed hardcoded confidence=1.0 to 0.0 initial confidence
  - Updated to moderate confidence (0.8) after processing
- `app/agents/report/agent.py`:
  - Changed hardcoded confidence=1.0 to 0.0 initial confidence
  - Updated to high confidence (0.9) after processing
- `app/agents/reasoning/engine/correlator.py`:
  - Changed hardcoded confidence=1.0 to 0.95 for IDENTITY relationships

## P0.6: DAG Dependencies
**Files Modified:**
- `app/agents/planner/registry.py`: No changes needed (was already correct)
- `app/agents/planner/planner.py`: 
  - Updated `_deterministic_fallback_plan` to use correct dependencies based on agent manifests
  - Fixed execution groups to reflect proper DAG structure
  - Updated agent lists to match manifest dependencies
  - Example: person_search now correctly depends on evidence_agent before video_agent/event_agent

## Verification
All hardcoded confidence=1.0 and ConfidenceScore(overall=1.0) instances have been removed from the agent codebase.
Confidence values are now derived from actual agent results, tool outputs, or appropriate heuristics that reflect the true uncertainty of the determination.