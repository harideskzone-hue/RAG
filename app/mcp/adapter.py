import logging
import time
from typing import Dict, Any
from uuid import uuid4

from app.schemas.context import VistaContext
from app.mcp.registry import ToolRegistry
from app.mcp.client import LocalMCPClient
from app.mcp.normalizer import MCPNormalizer

logger = logging.getLogger(__name__)

class MCPToolAdapter:
    """
    Acts as the boundary between the VISTA Dispatcher and external MCP tools.
    Handles Tool Validation, Pre-Execution RBAC, Execution, and Evidence Normalization.
    """
    
    def __init__(self):
        self.client = LocalMCPClient()
        self.normalizer = MCPNormalizer()
        
        # Execution limits
        self.MAX_MCP_CALLS_PER_REQUEST = 3
        self.MAX_CLIP_DURATION_SECONDS = 300 # 5 minutes

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: VistaContext) -> Any:
        """
        Executes a registered MCP tool. This is called directly by the Dispatcher,
        avoiding the creation of a new agent.
        """
        request_id = uuid4()
        start_time = time.time()
        
        # 0. Execution Limits
        if context.mcp_execution_count >= self.MAX_MCP_CALLS_PER_REQUEST:
            reason = f"MCP execution limit reached ({self.MAX_MCP_CALLS_PER_REQUEST})"
            self._log_audit(request_id, tool_name, arguments, context, "BLOCKED", reason, start_time)
            raise RuntimeError(reason)
        context.mcp_execution_count += 1
        
        # 1. Pydantic Schema Validation (Allowlist check)
        try:
            validated_args_obj = ToolRegistry.validate_tool_request(tool_name, arguments)
            validated_args = validated_args_obj.model_dump()
        except ValueError as e:
            self._log_audit(request_id, tool_name, arguments, context, "FAILED", str(e), start_time)
            raise ValueError(f"MCP Schema Validation failed: {e}")
            
        # 2. Pre-execution RBAC
        # For simplicity, if the tool arguments contain a camera_id, check it immediately.
        # This prevents unauthorized access to the underlying MCP resource.
        requested_camera = validated_args.get("camera_id")
        if requested_camera and context.user:
            if context.user.allowed_cameras is not None and requested_camera not in context.user.allowed_cameras:
                reason = f"Pre-execution RBAC blocked access to camera {requested_camera}"
                self._log_audit(request_id, tool_name, validated_args, context, "BLOCKED", reason, start_time)
                raise PermissionError(reason)
                
        # Specialized Pre-execution constraints
        if tool_name == "get_video_clip":
            from datetime import datetime
            try:
                start_dt = datetime.fromisoformat(validated_args["start_time"].replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(validated_args["end_time"].replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds()
                if duration > self.MAX_CLIP_DURATION_SECONDS or duration <= 0:
                    reason = f"Requested clip duration {duration}s invalid or exceeds maximum {self.MAX_CLIP_DURATION_SECONDS}s"
                    self._log_audit(request_id, tool_name, validated_args, context, "BLOCKED", reason, start_time)
                    raise ValueError(reason)
            except Exception as e:
                import traceback
                traceback.print_exc()
                reason = f"Invalid timestamp format or duration check failed: {e}"
                self._log_audit(request_id, tool_name, validated_args, context, "FAILED", reason, start_time)
                raise ValueError(reason)
                
        # 3. Tool Execution (UNTRUSTED RESPONSE)
        try:
            raw_response = await self.client.call_tool(tool_name, validated_args)
        except Exception as e:
            self._log_audit(request_id, tool_name, validated_args, context, "FAILED", f"Execution error: {e}", start_time)
            raise RuntimeError(f"MCP Tool execution failed: {e}")
            
        # 4. Response Schema Validation & Post-execution RBAC & Normalization
        try:
            if tool_name == "search_person_occurrences":
                normalized_evidence = self.normalizer.normalize_person_occurrences(
                    raw_response, 
                    context.user if context.user else None,
                    request_id
                )
            elif tool_name == "get_video_clip":
                normalized_evidence = self.normalizer.normalize_video_clip(
                    raw_response,
                    context,
                    request_id,
                    validated_args["evidence_id"]
                )
            elif tool_name == "search_vehicle_occurrences":
                normalized_evidence = self.normalizer.normalize_vehicle_occurrences(
                    raw_response, 
                    context.user if context.user else None,
                    request_id
                )
            elif tool_name == "search_alerts":
                normalized_evidence = self.normalizer.normalize_alerts(
                    raw_response, 
                    context.user if context.user else None,
                    request_id
                )
            elif tool_name == "get_camera_metadata":
                normalized_evidence = self.normalizer.normalize_camera_metadata(
                    raw_response, 
                    context.user if context.user else None,
                    request_id
                )
            else:
                # Fallback for future tools
                normalized_evidence = []
                
            # Append to the centralized EvidenceBundle!
            if not context.evidence_bundle:
                from app.domain.evidence import EvidenceBundle
                context.evidence_bundle = EvidenceBundle()
                
            context.evidence_bundle.evidence.extend(normalized_evidence)
            
            # Log success
            self._log_audit(
                request_id, 
                tool_name, 
                validated_args, 
                context, 
                "SUCCESS", 
                "None", 
                start_time,
                evidence_ids=[str(e.evidence_id) for e in normalized_evidence]
            )
            
            # Return something standard for the dispatcher (since it expects a result)
            from app.schemas.context import BaseResult
            from app.domain.models.confidence import ConfidenceScore
            return BaseResult(
                success=True,
                confidence=ConfidenceScore(overall=1.0) # EvidenceBundle handles real confidence
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log_audit(request_id, tool_name, validated_args, context, "FAILED", f"Normalization error: {e}", start_time)
            raise RuntimeError(f"MCP Evidence Normalization failed: {e}")

    def _log_audit(self, req_id, tool_name, args, context, status, reason, start_time, evidence_ids=None):
        duration = (time.time() - start_time) * 1000
        user_id = context.user.user_id if context.user else "system"
        logger.info(
            f"[MCP_AUDIT] RequestID={req_id} Tool={tool_name} User={user_id} "
            f"Args={args} Status={status} Duration={duration:.2f}ms Reason={reason} Evidence={evidence_ids}"
        )
