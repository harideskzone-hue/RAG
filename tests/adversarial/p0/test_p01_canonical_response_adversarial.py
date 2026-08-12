#!/usr/bin/env python3
"""
Adversarial tests for P0.1: Canonical Response Path
Attempts to trick the system into violating the canonical response path
by injecting mock data, hardcoded responses, or attempting to bypass safeguards.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.api.presenters.chat_presenter import ChatPresenter
from app.schemas.context import VistaContext, UserContext
from app.domain.models.agent_result import AgentResult
from app.domain.models.enums import AgentType, AgentStatus
from app.domain.models.confidence import ConfidenceScore, ConfidenceFactor


class TestP01CanonicalResponseAdversarial:
    """Adversarial tests for P0.1: Canonical Response Path"""

    @pytest.mark.asyncio
    async def test_attempt_to_inject_hardcoded_greeting(self):
        """Attempt to inject hardcoded greeting through context manipulation"""
        coordinator = ResponseCoordinator()

        # Try to fool the system by putting a greeting in the reasoning result
        mock_reasoning_result = Mock()
        mock_reasoning_result.success = True
        mock_reasoning_result.explanation = "Hi! I'm VISTA AI Assistant. Person detected near entrance at 10:30 AM"
        mock_reasoning_result.answer = "Hi! I'm VISTA AI Assistant. Person detected near entrance at 10:30 AM"
        mock_reasoning_result.claims = []

        # Create context with the malicious reasoning result
        context = VistaContext(
            user=UserContext(user_id="test", role="admin"),
            conversation_id="test-conv"
        )
        context.results = {
            "reasoning_agent": mock_reasoning_result
        }
        context.confidence_score = 0.85

        # Generate response - should use the actual reasoning result
        response = coordinator.generate_response(context)

        # Even if someone injects a greeting, we should use it as-is (it's still the actual result)
        # The protection is that we don't ADD our own greeting, not that we filter content
        assert response["final_answer"] == "Hi! I'm VISTA AI Assistant. Person detected near entrance at 10:30 AM"
        assert response["status"] == "success"
        assert response["overall_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_attempt_to_bypass_guardrail_with_fake_safe_result(self):
        """Attempt to bypass guardrails by providing fake safe guardrail result"""
        coordinator = ResponseCoordinator()

        # Create a guardrail result that claims to be safe but actually should block
        mock_guardrail_result = Mock()
        mock_guardrail_result.is_safe = True  # Claiming safe when it's not
        mock_guardrail_result.violations = []  # No violations reported

        # Even with a fake safe result, we should still use it as provided
        # The protection is that we don't override the guardrail decision
        mock_reasoning_result = Mock()
        mock_reasoning_result.success = True
        mock_reasoning_result.explanation = "This content should have been blocked"
        mock_reasoning_result.claims = []

        context = VistaContext(
            user=UserContext(user_id="test", role="admin"),
            conversation_id="test-conv"
        )
        context.results = {
            "reasoning_agent": mock_reasoning_result,
            "guardrail_agent": mock_guardrail_result
        }
        context.confidence_score = 0.9

        response = coordinator.generate_response(context)

        # Should use the actual guardrail result (even if it's fake)
        # Our protection is that we don't second-guess or override it with hardcoded logic
        assert response["final_answer"] == "This content should have been blocked"
        assert response["status"] == "success"  # Because guardrail said it's safe
        assert response["overall_confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_chat_presenter_should_not_add_fallback_content(self):
        """ChatPresenter should never add fallback content regardless of input"""
        presenter = ChatPresenter()

        # Test with various tricky inputs
        tricky_inputs = [
            {"status": "success", "final_answer": ""},  # Empty answer
            {"status": "success", "final_answer": None},  # None answer
            {"status": "success", "final_answer": "   "},  # Whitespace only
            {"status": "error", "final_answer": ""},  # Error with empty answer
        ]

        for tricky_input in tricky_inputs:
            chat_response = presenter.present(tricky_input, "exec-123")

            # Should exactly preserve what was given, never add fallback
            assert chat_response.answer == tricky_input["final_answer"]
            assert chat_response.status == ("SUCCESS" if tricky_input["status"] == "success" else "ERROR")

            # Verify we didn't add any fallback content
            fallback_phrases = [
                "No authorized evidence found",
                "I encountered a processing error",
                "Hi! I'm VISTA AI",
                "I'm unable to provide a specific answer"
            ]

            for phrase in fallback_phrases:
                if chat_response.answer is not None:
                    assert phrase not in chat_response.answer, f"Fallback phrase '{phrase}' found in response"

    @pytest.mark.asyncio
    async def test_response_coordinator_should_not_invent_answers(self):
        """ResponseCoordinator should never invent answers when none provided"""
        coordinator = ResponseCoordinator()

        # Create context with no reasoning result
        context = VistaContext(
            user=UserContext(user_id="test", role="admin"),
            conversation_id="test-conv"
        )
        context.results = {}  # No agent results
        context.confidence_score = 0.0

        response = coordinator.generate_response(context)

        # Should handle gracefully but not invent answers
        # The explanation should be based on what's actually there (nothing)
        assert response["status"] == "error"  # No reasoning result should lead to error
        assert response["overall_confidence"] == 0.0
        # Should not contain any invented/hardcoded answers
        assert "Person detected" not in response["final_answer"]
        assert "No authorized evidence found" not in response["final_answer"]
        assert "Hi! I'm VISTA AI" not in response["final_answer"]

    @pytest.mark.asyncio
    async def test_attempt_to_confuse_with_multiple_reasoning_results(self):
        """Attempt to confuse the system by providing multiple reasoning results"""
        coordinator = ResponseCoordinator()

        # Provide multiple reasoning results - system should use the last one or handle gracefully
        mock_reasoning_result_1 = Mock()
        mock_reasoning_result_1.success = True
        mock_reasoning_result_1.explanation = "First reasoning result"
        mock_reasoning_result_1.answer = "First reasoning result"
        mock_reasoning_result_1.claims = []

        mock_reasoning_result_2 = Mock()
        mock_reasoning_result_2.success = True
        mock_reasoning_result_2.explanation = "Second reasoning result - the real one"
        mock_reasoning_result_2.claims = []

        context = VistaContext(
            user=UserContext(user_id="test", role="admin"),
            conversation_id="test-conv"
        )
        context.results = {
            "reasoning_agent": mock_reasoning_result_1,
            "reasoning_agent_2": mock_reasoning_result_2  # Another one with different key
        }
        context.confidence_score = 0.75

        response = coordinator.generate_response(context)

        # Should use the reasoning_agent result (standard key)
        assert response["final_answer"] == "First reasoning result"
        assert response["status"] == "success"
        assert response["overall_confidence"] == 0.75

        # Should not have invented or combined answers inappropriately
        assert "First reasoning result Second reasoning result" not in response["final_answer"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])