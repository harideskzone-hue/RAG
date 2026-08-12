#!/usr/bin/env python3
"""
Tests for P0.6: DAG Dependencies
Validates that the planner uses correct DAG dependencies from agent manifests
and that execution groups reflect proper topological ordering.
"""
import pytest
from unittest.mock import Mock, patch
from app.agents.planner.agent import PlannerAgent
from app.agents.planner.planner import ExecutionPlanner
from app.schemas.context import VistaContext, UserContext


class TestP06DagDependencies:
    """Test P0.6: DAG Dependencies"""

    def test_planner_uses_correct_dependencies_from_manifests(self):
        """Planner should use dependencies defined in agent manifests, not hardcoded values"""
        planner = ExecutionPlanner()

        # Test that the deterministic fallback plan uses correct dependencies
        # This test verifies that our fix in _deterministic_fallback_plan is working

        # Intent: PERSON_SEARCH should require: intent_agent -> vector_agent -> evidence_agent
        # (based on agent manifests: vector_agent depends on evidence_agent)
        context = VistaContext(
            user=UserContext(user_id="test", role="admin")
        )
        context.execution_plan = Mock()
        context.execution_plan.agents = ["intent_agent", "vector_agent", "evidence_agent"]
        context.execution_plan.intent = "PERSON_SEARCH"

        # Get the fallback plan
        plan = planner._deterministic_fallback_plan(context)

        # Should have proper execution groups reflecting dependencies
        # Based on manifests:
        # - evidence_agent: no deps (can run first)
        # - vector_agent: depends on evidence_agent
        # - intent_agent: no deps (can run first)
        # So valid grouping could be: [intent_agent, evidence_agent] then [vector_agent]
        # OR [evidence_agent] then [intent_agent, vector_agent] if intent_agent doesn't depend on evidence

        # Check that we have reasonable execution groups
        assert len(plan.execution_groups) >= 1
        assert len(plan.execution_groups) <= 3  # Shouldn't need more than 3 groups for 3 agents

        # Flatten all agents in plan
        all_planned_agents = []
        for group in plan.execution_groups:
            all_planned_agents.extend(group.agents)

        # Should contain all the agents we asked for
        assert set(all_planned_agents) == set(["intent_agent", "vector_agent", "evidence_agent"])

        # Find the group index of each agent
        agent_group_indices = {}
        for i, group in enumerate(plan.execution_groups):
            for agent in group.agents:
                agent_group_indices[agent] = i

        # evidence_agent should run AFTER or in a later group than vector_agent
        assert agent_group_indices["vector_agent"] < agent_group_indices["evidence_agent"]

    def test_video_analysis_requires_metadata_first(self):
        """Video analysis should require metadata processing first"""
        planner = ExecutionPlanner()

        context = VistaContext(
            user=UserContext(user_id="test", role="admin")
        )
        context.execution_plan = Mock()
        context.execution_plan.agents = ["metadata_agent", "video_agent"]
        context.execution_plan.intent = "VIDEO_ANALYSIS"

        plan = planner._deterministic_fallback_plan(context)

        # Based on agent manifests:
        # - metadata_agent: no deps
        # - video_agent: depends on metadata_agent
        # So metadata_agent should run before video_agent

        # Check execution order
        agent_group_indices = {}
        for i, group in enumerate(plan.execution_groups):
            for agent in group.agents:
                agent_group_indices[agent] = i

        # metadata_agent should run before video_agent
        assert agent_group_indices["metadata_agent"] < agent_group_indices["video_agent"]

    def test_reasoning_requires_evidence_first(self):
        """Reasoning should require evidence processing first"""
        planner = ExecutionPlanner()

        context = VistaContext(
            user=UserContext(user_id="test", role="admin")
        )
        context.execution_plan = Mock()
        context.execution_plan.agents = ["evidence_agent", "reasoning_agent"]
        context.execution_plan.intent = "REASONING"

        plan = planner._deterministic_fallback_plan(context)

        # Based on agent manifests:
        # - evidence_agent: no deps (for basic metadata evidence)
        # - reasoning_agent: depends on evidence_agent (through evidence_bundle)
        # So evidence_agent should run before reasoning_agent

        agent_group_indices = {}
        for i, group in enumerate(plan.execution_groups):
            for agent in group.agents:
                agent_group_indices[agent] = i

        # reasoning_agent should run after evidence_agent
        assert agent_group_indices["evidence_agent"] < agent_group_indices["reasoning_agent"]

    def test_no_duplicate_agents_in_plan(self):
        """Planner should not include duplicate agents in execution plan"""
        planner = ExecutionPlanner()

        context = VistaContext(
            user=UserContext(user_id="test", role="admin")
        )
        context.execution_plan = Mock()
        context.execution_plan.agents = ["metadata_agent", "vector_agent", "metadata_agent"]  # duplicate
        context.execution_plan.intent = "TEST"

        plan = planner._deterministic_fallback_plan(context)

        # Count occurrences of each agent
        agent_counts = {}
        for group in plan.execution_groups:
            for agent in group.agents:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        # Each agent should appear exactly once
        for agent, count in agent_counts.items():
            assert count == 1, f"Agent {agent} appears {count} times in plan"

        # Should have exactly the unique agents we requested
        assert set(agent_counts.keys()) == set(["metadata_agent", "vector_agent"])

    def test_planner_handles_missing_dependencies_gracefully(self):
        """Planner should handle agents not in manifest registry gracefully"""
        from app.agents.intent.schemas import IntentResult
        from app.agents.intent.enums import Intent
        from app.domain.models.confidence import ConfidenceScore
        planner = ExecutionPlanner()
    
        intent_result = IntentResult(
            success=True,
            intent=Intent.UNKNOWN,
            confidence=ConfidenceScore(overall=1.0, factors=[]),
            entities={}
        )
    
        # Should not crash - should create a plan anyway
        plan = planner._deterministic_fallback_plan(intent_result)

        # Should still produce a valid plan
        assert plan is not None
        assert len(plan.execution_groups) >= 1

        # Should include both agents
        all_planned_agents = []
        for group in plan.execution_groups:
            all_planned_agents.extend(group.agents)
        assert set(all_planned_agents) == set(["metadata_agent"])

    @pytest.mark.asyncio
    async def test_planner_integration_with_context(self):
        """Test that planner works correctly with full context"""
        from app.agents.planner.planner import ExecutionPlanner
        from app.domain.models.confidence import ConfidenceScore
        planner = PlannerAgent(ExecutionPlanner(), Mock())

        context = VistaContext(
            user=UserContext(user_id="test", role="admin", allowed_cameras=["CAM_01", "CAM_02"])
        )
        context.current_query = "Find person in blue shirt"

        # Set intent via a proper IntentResult in context.results instead:
        from app.agents.intent.schemas import IntentResult
        from app.agents.intent.enums import Intent
        context.results["intent_agent"] = IntentResult(
            success=True,
            intent=Intent.PERSON_SEARCH,
            confidence=ConfidenceScore(overall=0.95, factors=[]),
            entities={"description": "person in blue shirt"}
        )

        # Execute the planner
        result = await planner.execute(context, None)

        # Should succeed
        assert result.success == True

        # Should have produced an execution plan
        assert context.execution_plan is not None

        # The execution plan should have proper execution groups
        assert hasattr(context.execution_plan, 'execution_groups')
        assert len(context.execution_plan.execution_groups) >= 1

        # Verify no hardcoded confidence in the result
        assert result.confidence.overall != 1.0
        assert isinstance(result.confidence, ConfidenceScore)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])