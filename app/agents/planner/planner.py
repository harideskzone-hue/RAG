import json

from app.agents.intent.schemas import IntentResult
from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from app.schemas.context import ExecutionPlan, ToolRequirement
from app.domain.models.enums import ExecutionMode


class ExecutionPlanner:
    """
    Transforms IntentResult into a deterministic ExecutionPlan using an LLM.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    async def plan(self, intent_result: IntentResult, query: str = "", execution_mode: ExecutionMode = ExecutionMode.ITERATIVE) -> ExecutionPlan | None:
        plan_obj = None
        if not self.llm or (intent_result.confidence >= 0.9 and intent_result.intent.value != "unknown"):
            # Fallback deterministic planning for MVP / Testing without LLM
            plan_obj = self._deterministic_fallback_plan(intent_result)
        else:
            try:
                response = await self.llm.ainvoke([
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": PLANNER_USER_PROMPT.format(
                        intent=intent_result.intent.value,
                        entities=intent_result.entities
                    )}
                ])

                import re
                content = response.content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                parsed = json.loads(content)

                tools = [ToolRequirement(**t) for t in parsed.get("tools", [])]

                plan_obj = ExecutionPlan(
                    success=True,
                    intent=parsed.get("intent", intent_result.intent.value),
                    agents=parsed.get("agents", []),
                    tools=tools,
                    execution_groups=parsed.get("execution_groups", []),
                    dependencies=parsed.get("dependencies", {}),
                    priority=parsed.get("priority", "normal"),
                    risk_level=parsed.get("risk_level", "LOW"),
                    requires_vlm=parsed.get("requires_vlm", False),
                    requires_confirmation=parsed.get("requires_confirmation", False),
                    estimated_tokens=parsed.get("estimated_tokens", 0),
                    estimated_latency_ms=parsed.get("estimated_latency_ms", 0),
                    estimated_tools=parsed.get("estimated_tools", len(tools)),
                    estimated_llm_calls=parsed.get("estimated_llm_calls", 0)
                )

                if not plan_obj.agents and plan_obj.intent != "greeting":
                    raise ValueError("LLM failed to schedule agents")
            except Exception as e:
                # On failure, fallback to deterministic rules
                import logging
                logging.getLogger(__name__).warning(
                    f"LLM planner failed, using deterministic fallback: {e}"
                )
                plan_obj = self._deterministic_fallback_plan(intent_result)

        # Run CostOptimizer if SIMPLE mode
        if execution_mode == ExecutionMode.SIMPLE:
            from app.agents.planner.optimizer import CostOptimizer
            optimizer = CostOptimizer()
            plan_obj = optimizer.optimize(plan_obj)

        # Decompose the plan into DAG tasks
        from app.agents.planner.decomposer import TaskDecomposer
        decomposer = TaskDecomposer(self.llm)
        plan_obj = await decomposer.decompose(query, plan_obj)

        return plan_obj

    def _deterministic_fallback_plan(self, intent_result_or_context) -> ExecutionPlan:
        """Accept either an IntentResult or a VistaContext for backward compatibility."""
        from app.agents.intent.schemas import IntentResult as _IntentResult
        from app.schemas.context import VistaContext as _VistaContext
        from app.schemas.context import ExecutionGroup

        if isinstance(intent_result_or_context, _VistaContext):
            # Called from tests with a VistaContext
            ctx = intent_result_or_context
            raw_intent = getattr(ctx.execution_plan, "intent", "") if ctx.execution_plan else ""
            agents_requested = getattr(ctx.execution_plan, "agents", []) if ctx.execution_plan else []
            intent = raw_intent.lower() if raw_intent else "unknown"
        else:
            # Original path: called with IntentResult
            intent_result = intent_result_or_context
            intent = intent_result.intent.value
            agents_requested = []

        return self._build_plan(intent, agents_requested)

    def _build_plan(self, intent: str, agents_requested: list[str]) -> ExecutionPlan:
        """Core deterministic planning logic."""
        from app.schemas.context import ExecutionGroup
        from app.agents.registry import (
            AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, 
            AGENT_EVENT, AGENT_REASONING, AGENT_EVIDENCE, AGENT_REPORT
        )

        # Default simple plan
        plan = ExecutionPlan(
            success=True,
            intent=intent,
            agents=[AGENT_METADATA],
            tools=[ToolRequirement(name="postgres", required=True)],
            execution_groups=[ExecutionGroup(agents=[AGENT_METADATA])],
            dependencies={},
            risk_level="LOW",
            estimated_tokens=50,
            estimated_latency_ms=100,
            estimated_tools=1,
            estimated_llm_calls=0
        )

        if intent == "person_search":
            plan.agents = [AGENT_METADATA, AGENT_VECTOR, AGENT_EVIDENCE, AGENT_VIDEO, AGENT_EVENT, AGENT_REASONING]
            plan.tools = [
                ToolRequirement(name="postgres", required=True),
                ToolRequirement(name="milvus", required=True),
                ToolRequirement(name="s3", required=False)
            ]
            plan.execution_groups = [
                ExecutionGroup(agents=[AGENT_METADATA, AGENT_VECTOR]),  # Layer 1: parallel
                ExecutionGroup(agents=[AGENT_EVIDENCE]),                  # Layer 2: depends on metadata and vector
                ExecutionGroup(agents=[AGENT_VIDEO, AGENT_EVENT]),      # Layer 3: depends on evidence
                ExecutionGroup(agents=[AGENT_REASONING])                  # Layer 4: depends on metadata, vector, event, video (per manifest)
            ]
            plan.dependencies = {
                AGENT_METADATA: [],
                AGENT_VECTOR: [],
                AGENT_EVIDENCE: [AGENT_METADATA, AGENT_VECTOR],
                AGENT_VIDEO: [AGENT_EVIDENCE],
                AGENT_EVENT: [AGENT_EVIDENCE],
                AGENT_REASONING: [AGENT_METADATA, AGENT_VECTOR, AGENT_EVENT, AGENT_VIDEO]
            }
            plan.requires_vlm = True
            plan.estimated_tokens = 6000
            plan.estimated_latency_ms = 2000
            plan.estimated_tools = 3
            plan.estimated_llm_calls = 2
        elif intent == "report":
            plan.agents = [AGENT_METADATA, AGENT_REPORT]
            plan.execution_groups = [ExecutionGroup(agents=[AGENT_METADATA]), ExecutionGroup(agents=[AGENT_REPORT])]
            plan.dependencies = {AGENT_REPORT: [AGENT_METADATA]}
            plan.estimated_tokens = 1000
            plan.estimated_latency_ms = 800
        elif intent == "greeting":
            plan.agents = []
            plan.tools = []
            plan.execution_groups = []
            plan.dependencies = {}
            plan.estimated_tokens = 0
            plan.estimated_latency_ms = 10
            plan.estimated_tools = 0

        # Override/filter if specific agents were requested
        if agents_requested:
            # We want unique agents keeping their order
            unique_requested = []
            for a in agents_requested:
                if a not in unique_requested:
                    unique_requested.append(a)
            
            plan.agents = unique_requested
            
            # Rebuild execution groups based on the requested agents
            # (Just a simple sequential grouping for tests, real optimizer handles DAG)
            new_groups = []
            
            # Simple heuristic: if we have a predefined plan for this intent, 
            # preserve the relative ordering from that predefined plan
            if plan.execution_groups and not intent == "unknown":
                for group in plan.execution_groups:
                    valid_agents = [a for a in group.agents if a in unique_requested]
                    if valid_agents:
                        new_groups.append(ExecutionGroup(agents=valid_agents))
                
                # Add any requested agents that weren't in the predefined plan at the end sequentially
                planned_agents = set(a for g in new_groups for a in g.agents)
                missing_agents = [a for a in unique_requested if a not in planned_agents]
                for a in missing_agents:
                    new_groups.append(ExecutionGroup(agents=[a]))
            else:
                # No predefined plan or "unknown" intent: just run them in sequence
                for a in unique_requested:
                    new_groups.append(ExecutionGroup(agents=[a]))
                    
            plan.execution_groups = new_groups

        return plan