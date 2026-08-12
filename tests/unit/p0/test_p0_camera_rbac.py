#!/usr/bin/env python3
"""
Tests for P0.3: Camera RBAC
Validates that camera-based Role-Based Access Control is enforced
at evidence-producing boundaries (S3 tool, video service, etc.)
"""
import pytest
from unittest.mock import Mock, AsyncMock
from app.tools.video.s3_tool import S3Tool
from app.services.video_service.service import VideoService
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import VistaContext, UserContext
from app.tools.video.schemas import VideoToolResult


class TestP03CameraRBAC:
    """Test P0.3: Camera RBAC"""

    @pytest.mark.asyncio
    async def test_s3_tool_enforces_camera_rbac(self):
        """S3Tool should check context.user.allowed_cameras and block unauthorized access"""
        event_bus = EventBus()
        s3_tool = S3Tool(event_bus)

        # Create context with restricted camera access
        user_context = UserContext(
            user_id="test-user",
            role="security",
            allowed_cameras=["CAM_01", "CAM_02"]  # Only these cameras allowed
        )
        context = VistaContext(user=user_context)

        # Try to access an unauthorized camera
        result = await s3_tool.execute(
            context,
            camera_id="CAM_03",  # NOT in allowed list
            start_time="2023-01-01T10:00:00Z",
            end_time="2023-01-01T10:05:00Z"
        )

        # Should fail with appropriate error
        assert result.success == False
        assert "not in allowed cameras" in result.error
        assert "CAM_03" in result.error
        assert "CAM_01" in result.error or "CAM_02" in result.error

    @pytest.mark.asyncio
    async def test_s3_tool_allows_authorized_camera(self):
        """S3Tool should allow access to cameras in the allowed list"""
        event_bus = EventBus()
        s3_tool = S3Tool(event_bus)

        # Mock the store.get_uri to avoid actual S3 calls
        original_get_uri = s3_tool.store.get_uri
        s3_tool.store.get_uri = AsyncMock(return_value=("http://example.com/video.mp4", 15.5))

        try:
            # Create context with camera access
            user_context = UserContext(
                user_id="test-user",
                role="security",
                allowed_cameras=["CAM_01", "CAM_02"]
            )
            context = VistaContext(user=user_context)

            # Try to access an authorized camera
            result = await s3_tool.execute(
                context,
                camera_id="CAM_01",  # IS in allowed list
                start_time="2023-01-01T10:00:00Z",
                end_time="2023-01-01T10:05:00Z"
            )

            # Should succeed
            assert result.success == True
            assert result.video_uri == "http://example.com/video.mp4"
            assert result.metadata["size_mb"] == 15.5
        finally:
            # Restore original method
            s3_tool.store.get_uri = original_get_uri

    @pytest.mark.asyncio
    async def test_video_service_enforces_camera_rbac(self):
        """VideoService should check camera access before processing"""
        event_bus = EventBus()

        # Mock dependencies
        mock_s3_tool = Mock()
        mock_vlm = Mock()
        video_service = VideoService(mock_s3_tool, mock_vlm, event_bus)

        # Mock the S3 tool response
        mock_s3_result = Mock()
        mock_s3_result.success = True
        mock_s3_result.video_uri = "http://example.com/video.mp4"
        mock_s3_tool.execute = AsyncMock(return_value=mock_s3_result)

        # Mock VLM response
        mock_vlm_response = {
            "scene_summary": "Person detected",
            "objects": [],
            "activities": [],
            "confidence": 0.8,
            "frames_analyzed": 10,
            "timeline": [],
            "reasoning": "Test reasoning"
        }
        mock_vlm.analyze = AsyncMock(return_value=mock_vlm_response)

        # Mock other service dependencies
        video_service.clip_selector = Mock()
        video_service.clip_selector.select_clip_window = Mock(return_value=(Mock(), Mock()))
        video_service.cache = Mock()
        video_service.cache.get = Mock(return_value=None)  # Force cache miss
        video_service.cache.set = Mock()
        video_service.sampler = Mock()
        video_service.sampler.sample_frames = Mock(return_value=[])
        video_service.preprocessor = Mock()
        video_service.preprocessor.preprocess_frames = Mock(return_value=[])

        # Create context with RESTRICTED camera access
        user_context = UserContext(
            user_id="test-user",
            role="security",
            allowed_cameras=["CAM_01", "CAM_02"]  # CAM_03 is NOT allowed
        )
        from app.domain.models.confidence import ConfidenceReport
        from app.schemas.context import ReasoningContext

        reasoning_context = ReasoningContext(
            query="What is happening?",
            user=user_context
        )

        # Try to process video for UNAUTHORIZED camera
        # This should raise an ValueError due to RBAC check
        import asyncio
        try:
            await video_service.analyze_event(
                camera_id="CAM_03",  # NOT in allowed list
                timestamp=Mock(),
                context=reasoning_context
            )
            # If we reach here, the test failed - it should have raised an exception
            assert False, "Expected ValueError for unauthorized camera access"
        except ValueError as e:
            # Verify it's the right kind of error
            assert "not in allowed cameras" in str(e)
            assert "CAM_03" in str(e)
        except Exception as e:
            # Some other unexpected error
            assert False, f"Unexpected error type: {type(e)} - {e}"

    @pytest.mark.asyncio
    async def test_video_service_allows_authorized_camera(self):
        """VideoService should allow processing for authorized cameras"""
        import os
        from unittest.mock import Mock, AsyncMock, patch
        
        with patch.dict(os.environ, {"VISTA_ENV": "test", "VLM_PROVIDER": "gemini"}):
            from app.services.video_service.service import VideoService
            
        from app.graph.supervisor.event_bus import EventBus
        event_bus = EventBus()

        # Mock S3 Tool
        mock_s3 = AsyncMock()
        mock_s3_result = Mock()
        mock_s3_result.success = True
        mock_s3_result.video_uri = "http://example.com/video.mp4"
        mock_s3.execute = AsyncMock(return_value=mock_s3_result)
        
        # Mock VLM Provider
        mock_vlm = AsyncMock()
        mock_vlm_response = {
            "scene_summary": "Authorized camera scene",
            "objects": [],
            "activities": [],
            "confidence": 0.9,
            "frames_analyzed": 10,
            "timeline": [],
            "reasoning": "Test reasoning"
        }
        mock_vlm.analyze = AsyncMock(return_value=mock_vlm_response)
        
        video_service = VideoService(mock_s3, mock_vlm, event_bus)

        # Mock other service dependencies
        from datetime import datetime, timedelta
        start_dt = datetime.now()
        end_dt = start_dt + timedelta(seconds=10)
        video_service.clip_selector = Mock()
        video_service.clip_selector.select_clip_window = Mock(return_value=(start_dt, end_dt))
        video_service.cache = Mock()
        video_service.cache.get = Mock(return_value=None)
        video_service.cache.set = Mock()
        video_service.sampler = Mock()
        video_service.sampler.sample_frames = Mock(return_value=[])
        video_service.preprocessor = Mock()
        video_service.preprocessor.preprocess_frames = Mock(return_value=[])

        # Create context with RESTRICTED camera access
        user_context = UserContext(
            user_id="test-user",
            role="security",
            allowed_cameras=["CAM_01", "CAM_02"]
        )
        from app.schemas.context import ReasoningContext

        reasoning_context = ReasoningContext(
            query="What is happening?",
            user=user_context
        )

        # Try to process video for AUTHORIZED camera
        result = await video_service.analyze_event(
            camera_id="CAM_01",
            timestamp=Mock(),
            context=reasoning_context
        )
        
        assert result.get("scene_summary") == "Authorized camera scene"
        assert result.get("confidence") == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])