import json
import asyncio
from typing import Dict, Any
from uuid import uuid4

class FakeMCPClient:
    """
    Simulates a local MCP server that executes VISTA-specific tools.
    This returns raw, untrusted JSON dictionaries (as an external server would).
    """
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates calling an MCP tool over the network.
        Includes a simulated delay and returns an untrusted response.
        """
        await asyncio.sleep(0.1) # Simulate network latency
        
        if tool_name == "search_person_occurrences":
            return self._mock_search_person_occurrences(arguments)
        elif tool_name == "get_video_clip":
            return self._mock_get_video_clip(arguments)
        elif tool_name == "search_vehicle_occurrences":
            return self._mock_search_vehicle_occurrences(arguments)
        elif tool_name == "search_alerts":
            return self._mock_search_alerts(arguments)
        elif tool_name == "get_camera_metadata":
            return self._mock_get_camera_metadata(arguments)
        else:
            return {"status": "error", "message": f"Tool {tool_name} not implemented in local MCP server"}

    def _mock_search_vehicle_occurrences(self, arguments: dict[str, Any]) -> Dict[str, Any]:
        desc = arguments.get("description", "unknown")
        camera_id = arguments.get("camera_id", "")
        return {
            "status": "success",
            "results": [
                {
                    "evidence_id": str(uuid4()),
                    "camera_id": camera_id,
                    "timestamp": "2026-08-08T10:45:00Z",
                    "description": f"Vehicle matching '{desc}'",
                    "confidence": 0.88
                }
            ]
        }

    def _mock_search_alerts(self, arguments: dict[str, Any]) -> Dict[str, Any]:
        desc = arguments.get("description", "unknown")
        camera_id = arguments.get("camera_id", "")
        return {
            "status": "success",
            "results": [
                {
                    "evidence_id": str(uuid4()),
                    "camera_id": camera_id,
                    "timestamp": "2026-08-08T10:50:00Z",
                    "description": f"Alert matching '{desc}'",
                    "event_type": "security_alert",
                    "confidence": 0.95
                }
            ]
        }

    def _mock_get_camera_metadata(self, arguments: dict[str, Any]) -> Dict[str, Any]:
        camera_id = arguments.get("camera_id", "")
        return {
            "status": "success",
            "result": {
                "camera_id": camera_id,
                "location": "North Gate Entrance",
                "status": "active"
            }
        }

    def _mock_get_video_clip(self, arguments: dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "result": {
                "camera_id": arguments.get("camera_id"),
                "start_time": arguments.get("start_time"),
                "end_time": arguments.get("end_time"),
                "evidence_id": arguments.get("evidence_id"),
                "clip_uri": f"s3://vista-storage/clips/{arguments.get('camera_id')}_{arguments.get('start_time')}.mp4",
                "description": "Video clip containing the occurrence"
            }
        }

    def _mock_search_person_occurrences(self, arguments: dict[str, Any]) -> Dict[str, Any]:
        desc = arguments.get("description", "unknown")
        
        # Simulate returning a raw JSON payload with a mix of valid and invalid data
        # Notice we are generating raw IDs, ignoring camera_id requirements, etc.
        # This forces the adapter/normalizer to enforce the contract!
        return {
            "status": "success",
            "results": [
                {
                    "evidence_id": str(uuid4()), 
                    "camera_id": "CAM_01",
                    "timestamp": "2026-08-08T10:42:15Z",
                    "description": f"Match for {desc}",
                    "confidence": 0.85
                },
                {
                    "evidence_id": str(uuid4()), 
                    "camera_id": "CAM_02", # Might be unauthorized
                    "timestamp": "2026-08-08T10:45:00Z",
                    "description": f"Another match for {desc}",
                    "confidence": 0.75
                }
            ]
        }
