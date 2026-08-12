import logging
from typing import Dict, Any, List
from uuid import UUID
from datetime import datetime, timezone

from app.schemas.context import UserContext
from app.domain.evidence import PersonEvidence, EvidenceType
from app.domain.models.enums import EntityType

logger = logging.getLogger(__name__)

class MCPNormalizer:
    """
    Normalizes raw, untrusted MCP tool JSON output into strongly typed Evidence objects.
    Enforces post-execution RBAC and strict provenance tracking.
    """
    
    @staticmethod
    def normalize_person_occurrences(
        raw_response: Dict[str, Any], 
        user_context: UserContext,
        request_id: UUID
    ) -> List[PersonEvidence]:
        """
        Validates untrusted MCP JSON against Schema, RBAC, and Provenance rules.
        Returns a list of standardized PersonEvidence objects to be appended to the EvidenceBundle.
        """
        if raw_response.get("status") != "success":
            raise ValueError(f"MCP tool failed: {raw_response.get('message', 'Unknown error')}")
            
        results = raw_response.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Malformed response: 'results' must be a list")
            
        normalized_evidence = []
        for item in results:
            # 1. Schema / Provenance Validation
            evidence_id_str = item.get("evidence_id")
            camera_id = item.get("camera_id")
            timestamp_str = item.get("timestamp")
            description = item.get("description")
            
            if not evidence_id_str or not camera_id or not timestamp_str or not description:
                logger.warning(f"[{request_id}] Dropping MCP result due to missing provenance: {item}")
                continue
                
            try:
                evidence_id = UUID(evidence_id_str)
            except ValueError:
                logger.warning(f"[{request_id}] Dropping MCP result due to fake evidence_id: {evidence_id_str}")
                continue
                
            # Parse timestamp safely
            try:
                # Handle isoformat
                if timestamp_str.endswith('Z'):
                    timestamp = datetime.fromisoformat(timestamp_str[:-1])
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except Exception as e:
                logger.warning(f"[{request_id}] Dropping MCP result due to malformed timestamp: {timestamp_str}")
                continue
                
            # 2. Post-execution RBAC
            if user_context.allowed_cameras is not None and camera_id not in user_context.allowed_cameras:
                logger.warning(f"[{request_id}] Dropping MCP result for unauthorized camera: {camera_id}")
                continue
                
            # 3. Normalization
            normalized_evidence.append(
                PersonEvidence(
                    evidence_id=evidence_id,
                    evidence_type=EvidenceType.VECTOR,
                    source="mcp_search_person_occurrences",
                    confidence=float(item.get("confidence", 0.5)),
                    timestamp=timestamp,
                    trace_id=request_id,
                    metadata={"camera_id": camera_id, "description": description},
                    provenance={"request_id": str(request_id), "mcp_tool": "search_person_occurrences"}
                )
            )
            
        return normalized_evidence

    @staticmethod
    def normalize_video_clip(
        raw_response: Dict[str, Any],
        context: Any,
        request_id: UUID,
        requested_evidence_id: str
    ) -> List[Any]:
        from app.domain.evidence import VideoEvidence
        from uuid import uuid4
        
        if raw_response.get("status") != "success":
            raise ValueError(f"MCP tool failed: {raw_response.get('message', 'Unknown error')}")
            
        result = raw_response.get("result")
        if not result or not isinstance(result, dict):
            raise ValueError("Malformed response: 'result' must be a dict")
            
        camera_id = result.get("camera_id")
        clip_uri = result.get("clip_uri")
        evidence_id_str = result.get("evidence_id")
        
        if not camera_id or not clip_uri or not evidence_id_str:
            raise ValueError(f"[{request_id}] Dropping MCP result due to missing provenance in get_video_clip")
            
        if evidence_id_str != requested_evidence_id:
            raise ValueError(f"[{request_id}] Dropping MCP result: Returned evidence_id {evidence_id_str} does not match requested {requested_evidence_id}")
            
        # Post-execution RBAC
        user_context = context.user
        if user_context and user_context.allowed_cameras is not None and camera_id not in user_context.allowed_cameras:
            raise PermissionError(f"[{request_id}] Dropping MCP result for unauthorized camera: {camera_id}")
            
        # Validate that this evidence ID actually exists in our bundle and matches the camera!
        # This prevents the LLM from making up an evidence ID and bypassing constraints.
        if not context.evidence_bundle:
            raise ValueError(f"[{request_id}] No EvidenceBundle found in context to validate against")
            
        found = False
        for ev in context.evidence_bundle.evidence:
            if str(ev.evidence_id) == evidence_id_str:
                if ev.metadata.get("camera_id") != camera_id:
                    raise ValueError(f"[{request_id}] Camera ID mismatch: Evidence says {ev.metadata.get('camera_id')}, clip says {camera_id}")
                found = True
                break
                
        if not found:
            raise ValueError(f"[{request_id}] Requested evidence_id {evidence_id_str} not found in authoritative EvidenceBundle")
            
        return [
            VideoEvidence(
                evidence_id=uuid4(), # The clip itself gets a new UUID, but points back via provenance
                evidence_type=EvidenceType.VIDEO,
                source="mcp_get_video_clip",
                confidence=1.0, # Deterministic retrieval
                timestamp=datetime.now(),
                trace_id=request_id,
                metadata={
                    "camera_id": camera_id,
                    "description": result.get("description", "Video clip"),
                    "clip_uri": clip_uri,
                    "start_time": result.get("start_time"),
                    "end_time": result.get("end_time")
                },
                provenance={
                    "request_id": str(request_id),
                    "mcp_tool": "get_video_clip",
                    "occurrence_evidence_id": evidence_id_str
                }
            )
        ]

    @staticmethod
    def normalize_vehicle_occurrences(
        raw_response: Dict[str, Any], 
        user_context: UserContext,
        request_id: UUID
    ) -> List[Any]: # List[VehicleEvidence]
        from app.domain.evidence import VehicleEvidence
        
        if raw_response.get("status") != "success":
            raise ValueError(f"MCP tool failed: {raw_response.get('message', 'Unknown error')}")
            
        results = raw_response.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Malformed response: 'results' must be a list")
            
        normalized_evidence = []
        for item in results:
            evidence_id_str = item.get("evidence_id")
            camera_id = item.get("camera_id")
            timestamp_str = item.get("timestamp")
            description = item.get("description")
            
            if not evidence_id_str or not camera_id or not timestamp_str or not description:
                logger.warning(f"[{request_id}] Dropping MCP result due to missing provenance: {item}")
                continue
                
            try:
                evidence_id = UUID(evidence_id_str)
            except ValueError:
                logger.warning(f"[{request_id}] Dropping MCP result due to fake evidence_id: {evidence_id_str}")
                continue
                
            try:
                if timestamp_str.endswith('Z'):
                    timestamp = datetime.fromisoformat(timestamp_str[:-1])
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except Exception as e:
                logger.warning(f"[{request_id}] Dropping MCP result due to invalid timestamp: {e}")
                continue
                
            if user_context.allowed_cameras is not None and camera_id not in user_context.allowed_cameras:
                logger.warning(f"[{request_id}] Post-execution RBAC filtered unauthorized camera: {camera_id}")
                continue
                
            normalized_evidence.append(
                VehicleEvidence(
                    evidence_id=evidence_id,
                    evidence_type=EvidenceType.VECTOR,
                    source="mcp_search_vehicle_occurrences",
                    timestamp=timestamp,
                    confidence=float(item.get("confidence", 0.0)),
                    trace_id=request_id,
                    metadata={
                        "camera_id": camera_id,
                        "description": description
                    },
                    provenance={
                        "request_id": str(request_id),
                        "mcp_tool": "search_vehicle_occurrences"
                    }
                )
            )
        return normalized_evidence

    @staticmethod
    def normalize_alerts(
        raw_response: Dict[str, Any], 
        user_context: UserContext,
        request_id: UUID
    ) -> List[Any]: # List[EventEvidence]
        from app.domain.evidence import EventEvidence
        
        if raw_response.get("status") != "success":
            raise ValueError(f"MCP tool failed: {raw_response.get('message', 'Unknown error')}")
            
        results = raw_response.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Malformed response: 'results' must be a list")
            
        normalized_evidence = []
        for item in results:
            evidence_id_str = item.get("evidence_id")
            camera_id = item.get("camera_id")
            timestamp_str = item.get("timestamp")
            description = item.get("description")
            
            if not evidence_id_str or not camera_id or not timestamp_str or not description:
                logger.warning(f"[{request_id}] Dropping MCP result due to missing provenance: {item}")
                continue
                
            try:
                evidence_id = UUID(evidence_id_str)
            except ValueError:
                logger.warning(f"[{request_id}] Dropping MCP result due to fake evidence_id: {evidence_id_str}")
                continue
                
            try:
                if timestamp_str.endswith('Z'):
                    timestamp = datetime.fromisoformat(timestamp_str[:-1])
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except Exception as e:
                logger.warning(f"[{request_id}] Dropping MCP result due to invalid timestamp: {e}")
                continue
                
            if user_context.allowed_cameras is not None and camera_id not in user_context.allowed_cameras:
                logger.warning(f"[{request_id}] Post-execution RBAC filtered unauthorized camera: {camera_id}")
                continue
                
            normalized_evidence.append(
                EventEvidence(
                    evidence_id=evidence_id,
                    evidence_type=EvidenceType.EVENT,
                    source="mcp_search_alerts",
                    timestamp=timestamp,
                    confidence=float(item.get("confidence", 0.0)),
                    trace_id=request_id,
                    metadata={
                        "camera_id": camera_id,
                        "description": description,
                        "event_type": item.get("event_type", "alert")
                    },
                    provenance={
                        "request_id": str(request_id),
                        "mcp_tool": "search_alerts"
                    }
                )
            )
        return normalized_evidence

    @staticmethod
    def normalize_camera_metadata(
        raw_response: Dict[str, Any], 
        user_context: UserContext,
        request_id: UUID
    ) -> List[Any]: # List[MetadataEvidence]
        from app.domain.evidence import MetadataEvidence
        from uuid import uuid4
        
        if raw_response.get("status") != "success":
            raise ValueError(f"MCP tool failed: {raw_response.get('message', 'Unknown error')}")
            
        result = raw_response.get("result", {})
        camera_id = result.get("camera_id")
        
        if not camera_id:
            logger.warning(f"[{request_id}] Dropping MCP result due to missing provenance: {result}")
            return []
            
        if user_context.allowed_cameras is not None and camera_id not in user_context.allowed_cameras:
            logger.warning(f"[{request_id}] Post-execution RBAC filtered unauthorized camera: {camera_id}")
            return []
            
        return [
            MetadataEvidence(
                evidence_id=uuid4(),
                evidence_type=EvidenceType.METADATA,
                source="mcp_get_camera_metadata",
                timestamp=datetime.now(timezone.utc),
                confidence=1.0,
                trace_id=request_id,
                metadata={
                    "camera_id": camera_id,
                    "location": result.get("location", "unknown"),
                    "status": result.get("status", "unknown")
                },
                provenance={
                    "request_id": str(request_id),
                    "mcp_tool": "get_camera_metadata"
                }
            )
        ]
