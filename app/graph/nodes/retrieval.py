from typing import Dict, Any

from app.tools.db_tools import EvidenceSearchTool, TimelineTool, TrackSearchTool
from app.services.db_services import EvidenceService, TrackService

class RetrievalNode:
    """
    Executes DB tools based on the planner's deterministic output.
    Uses the actual active_video_id from execution state — never a placeholder.
    """
    def __init__(self, evidence_service: EvidenceService, track_service: TrackService):
        self.evidence_tool = EvidenceSearchTool(evidence_service)
        self.timeline_tool = TimelineTool(evidence_service)
        self.track_tool = TrackSearchTool(track_service)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if "abstain_reason" in state and state["abstain_reason"]:
            state["retrieved_evidence"] = []
            return state
            
        plan = state.get("execution_plan", [])
        intent = state.get("query_intent")
        
        # Active video ID must come from the execution context — never hardcoded
        video_id = state.get("active_video_id")
        if not video_id:
            state["abstain_reason"] = "No active video context. Cannot retrieve evidence without a video scope."
            state["retrieved_evidence"] = []
            return state
        
        results = []
        
        for tool_name in plan:
            if tool_name == "EvidenceSearchTool":
                res = await self.evidence_tool.execute(
                    context=None,
                    video_id=video_id,
                    camera_id=intent.spatial_constraints.camera_ids[0] if intent.spatial_constraints and intent.spatial_constraints.camera_ids else None,
                    track_id=intent.identity_target
                )
                results.append({"tool": "EvidenceSearchTool", "data": res})
                
            elif tool_name == "TimelineTool":
                res = await self.timeline_tool.execute(
                    context=None,
                    video_id=video_id,
                    person_id=intent.identity_target
                )
                results.append({"tool": "TimelineTool", "data": res})
                
            elif tool_name == "PersonSearchTool":
                res = await self.track_tool.execute(
                    context=None,
                    video_id=video_id,
                    track_id=intent.identity_target
                )
                results.append({"tool": "PersonSearchTool", "data": res})
                
        state["retrieved_evidence"] = results
        return state
