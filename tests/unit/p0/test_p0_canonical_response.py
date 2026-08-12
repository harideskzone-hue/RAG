#!/usr/bin/env python3
"""
Tests for P0.1: Canonical Response Path
Validates that responses follow the path:
ReasoningResult → GuardrailResult → ResponseCoordinator → ChatPresenter → API/UI
without hardcoded/mock answers.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.api.presenters.chat_presenter import ChatPresenter
from app.schemas.context import VistaContext, UserContext
from app.domain.models.agent_result import AgentResult
from app.domain.models.enums import AgentType, AgentStatus
from app.domain.models.confidence import ConfidenceScore, ConfidenceFactor


class TestP01CanonicalResponse:
    """Test P0.1: Canonical Response Path"""

    @pytest.mark.asyncio
    async def test_response_coordinator_uses_reasoning_result(self):
        """ResponseCoordinator should use actual ReasoningResult, not hardcoded answers"""
        coordinator = ResponseCoordinator()

        # Create a mock ReasoningResult with actual content
        mock_reasoning_result = Mock()
        mock_reasoning_result.success = True
        mock_reasoning_result.explanation = "Person detected near entrance at 10:30 AM"
        mock_reasoning_result.answer = "Person detected near entrance at 10:30 AM"
        mock_reasoning_result.claims = []
        mock_reasoning_result.metadata = {}

        # Create context with the reasoning result
        context = VistaContext(
            user=UserContext(user_id="test", role="admin"),
            conversation_id="test-conv"
        )
        context.results = {
            "reasoning_agent": mock_reasoning_result
        }
        context.confidence_score = 0.85

        # Generate response
        response = coordinator.generate_response(context)

        # Verify it uses the actual reasoning result, not hardcoded text
        assert response["final_answer"] == "Person detected near entrance at 10:30 AM"
        assert response["status"] == "success"
        assert response["overall_confidence"] == 0.85

        # Verify no hardcoded greeting or mock responses
        assert "Hi! I'm VISTA AI" not in response["final_answer"]
        assert "I encountered a processing error" not in response["final_answer"]
        assert "No authorized evidence found" not in response["final_answer"]

    @pytest.mark.asyncio
    async def test_response_coordinator_handles_guardrail_block(self):
        """ResponseCoordinator should properly handle GuardrailResult blocking"""
        coordinator = ResponseCoordinator()

        # Create a mock GuardrailResult that blocks the response
        mock_guardrail_result = Mock()
        mock_guardrail_result.is_safe = False
        mock_guardrail_result.violations = ["violence", "harassment"]

        # Create context with the guardrail result
        context = VistaContext(
            user=UserContext(user_id="test", role="admin"),
            conversation_id="test-conv"
        )
        context.results = {
            "guardrail_agent": mock_guardrail_result
        }
        context.confidence_score = 0.9

        # Generate response
        response = coordinator.generate_response(context)

        # Verify it properly blocks and mentions violations
        assert "Response blocked by safety guardrails:" in response["final_answer"]
        assert "violence" in response["final_answer"]
        assert "harassment" in response["final_answer"]
        assert response["status"] == "error"  # or however blocking is represented

    @pytest.mark.asyncio
    async def test_chat_presenter_only_formats_without_adding_content(self):
        """ChatPresenter should only format/serialize, not add reasoning or fallback answers"""
        presenter = ChatPresenter()

        # Test with actual canonical response
        canonical_response = {
            "status": "success",
            "final_answer": "Person wearing blue shirt detected at loading dock",
            "overall_confidence": 0.78,
            "citations": [],
            "evidence": [],
            "processing_time_ms": 150
        }

        chat_response = presenter.present(canonical_response, "exec-123")

        # Verify it preserves the exact answer from canonical response
        assert chat_response.answer == "Person wearing blue shirt detected at loading dock"
        assert chat_response.status == "SUCCESS"
        assert chat_response.confidence == 0.78

        # Verify it doesn't add fallback content when answer is provided
        empty_canonical = {
            "status": "success",
            "final_answer": "",  # Empty but explicitly provided
            "overall_confidence": 0.0,
            "citations": [],
            "evidence": [],
            "processing_time_ms": 100
        }

        chat_response_empty = presenter.present(empty_canonical, "exec-124")
        assert chat_response_empty.answer == ""  # Should preserve empty string, not add fallback

    @pytest.mark.asyncio
    async def test_chat_presenter_handles_error_status_correctly(self):
        """ChatPresenter should properly format error responses from canonical response"""
        presenter = ChatPresenter()

        canonical_response = {
            "status": "error",
            "final_answer": "Database connection timeout",
            "error": "DB_CONN_TIMEOUT",
            "overall_confidence": 0.0,
            "citations": [],
            "evidence": [],
            "processing_time_ms": 2000
        }

        chat_response = presenter.present(canonical_response, "exec-125")

        # Should preserve the error information
        assert "Database connection timeout" in chat_response.answer
        assert chat_response.status == "ERROR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])