import json
from typing import Any

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
        # If LLM is not available or intent is highly confident, use deterministic fallback
        conf_val = getattr(intent_result.confidence, 'overall', intent_result.confidence) if hasattr(intent_result, 'confidence') else 0.0
        plan_obj = None
        if not self.llm:
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

                tools = []
                for t in parsed.get("tools", []):
                    if isinstance(t, dict):
                        if "name" not in t and "tool_name" in t:
                            t["name"] = t["tool_name"]
                        if "name" not in t and "tool" in t:
                            t["name"] = t["tool"]
                        if "name" not in t:
                            t["name"] = "unknown_tool"
                        if "required" in t and isinstance(t["required"], (list, dict)):
                            t["required"] = bool(t["required"])
                        if "arguments" in t and isinstance(t["arguments"], list):
                            t["arguments"] = {}
                        tools.append(ToolRequirement(**t))
                    elif isinstance(t, str):
                        tools.append(ToolRequirement(name=t))

                raw_deps = parsed.get("dependencies", {})
                if isinstance(raw_deps, list):
                    deps = {}
                    for item in raw_deps:
                        if isinstance(item, dict) and "agent" in item and "dependencies" in item:
                            deps[item["agent"]] = item.get("dependencies", [])
                    raw_deps = deps
                    
                # Normalize agents: LLM may return list of dicts instead of list of strings
                raw_agents = parsed.get("agents", [])
                normalized_agents = []
                for a in raw_agents:
                    if isinstance(a, dict):
                        normalized_agents.append(a.get("name", str(a)))
                    else:
                        normalized_agents.append(str(a))

                # Normalize execution_groups: LLM may return list-of-lists instead of list of {agents:[...]}
                from app.schemas.context import ExecutionGroup
                raw_groups = parsed.get("execution_groups", [])
                normalized_groups = []
                for g in raw_groups:
                    if isinstance(g, list):
                        # Flatten: list of agent name lists
                        flat = []
                        for item in g:
                            if isinstance(item, list):
                                flat.extend(str(x) for x in item)
                            else:
                                flat.append(str(item))
                        normalized_groups.append(ExecutionGroup(agents=flat))
                    elif isinstance(g, dict):
                        agent_list = g.get("agents", [])
                        normalized_groups.append(ExecutionGroup(agents=[str(x) for x in agent_list]))
                    elif isinstance(g, ExecutionGroup):
                        normalized_groups.append(g)

                # Normalize intent — LLM may return a dict instead of a string
                raw_intent = parsed.get("intent", intent_result.intent.value)
                if isinstance(raw_intent, dict):
                    raw_intent = raw_intent.get("name", intent_result.intent.value)
                normalized_intent = str(raw_intent)

                plan_obj = ExecutionPlan(
                    success=True,
                    intent=normalized_intent,
                    agents=normalized_agents,
                    tools=tools,
                    execution_groups=normalized_groups,
                    dependencies=raw_deps,
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

        return self._build_plan(intent, agents_requested, intent_result=intent_result_or_context if hasattr(intent_result_or_context, 'query_intent') else None)

    def _build_plan(self, intent: str, agents_requested: list[str], intent_result: Any = None) -> ExecutionPlan:
        """Core deterministic planning logic.
        
        Locked pipeline ordering:
            Layer 1: ALL retrieval agents (parallel)
            Layer 2: evidence_agent (normalization)
            Layer 3: evidence_fusion_agent (fusion, dedup, provenance)
            Layer 4: verification_agent (constraints → contract)
            Layer 5: reasoning_agent (LLM verbalization)
        """
        from app.schemas.context import ExecutionGroup
        from app.agents.registry import (
            AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, 
            AGENT_EVENT, AGENT_REASONING, AGENT_EVIDENCE, AGENT_REPORT,
            AGENT_FUSION, AGENT_VERIFICATION,
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

        query_intent = getattr(intent_result, 'query_intent', None)
        search_ops = getattr(query_intent, 'search_operations', []) if query_intent else []
        domain = getattr(query_intent, 'domain', 'investigation')

        # Capability-based plan selection
        if intent in ["time_query", "time"] or "time_query" in search_ops:
            from app.agents.registry import AGENT_TIME
            plan.agents = [AGENT_TIME]
            plan.tools = [ToolRequirement(name="time_tool", required=True)]
            plan.execution_groups = [ExecutionGroup(agents=[AGENT_TIME])]
            plan.dependencies = {AGENT_TIME: []}
            plan.estimated_tokens = 5
            plan.estimated_latency_ms = 5
            plan.estimated_tools = 1
            plan.estimated_llm_calls = 0
        elif intent in ["greeting", "capability_explanation", "general_query"] or domain == "general":
            from app.agents.registry import AGENT_REASONING
            plan.agents = [AGENT_REASONING]
            plan.tools = []
            plan.execution_groups = [ExecutionGroup(agents=[AGENT_REASONING])]
            plan.dependencies = {AGENT_REASONING: []}
            plan.estimated_tokens = 50
            plan.estimated_latency_ms = 50
            plan.estimated_tools = 0
        elif intent in ["count", "list"] or getattr(query_intent, 'operation', '') in ["count", "list"]:
            # Count/list queries: retrieval → normalization → fusion → verification → reasoning
            # Reasoning is REQUIRED — the LLM verbalizes the verified count, it doesn't calculate.
            plan.agents = [AGENT_METADATA, AGENT_VECTOR, AGENT_EVIDENCE, AGENT_FUSION, AGENT_VERIFICATION, AGENT_REASONING]
            plan.tools = [
                ToolRequirement(name="postgres", required=True),
                ToolRequirement(name="milvus", required=True)
            ]
            plan.execution_groups = [
                ExecutionGroup(agents=[AGENT_METADATA, AGENT_VECTOR]),  # Layer 1: retrieval
                ExecutionGroup(agents=[AGENT_EVIDENCE]),                 # Layer 2: normalization
                ExecutionGroup(agents=[AGENT_FUSION]),                   # Layer 3: fusion + dedup
                ExecutionGroup(agents=[AGENT_VERIFICATION]),             # Layer 4: verification → contract
                ExecutionGroup(agents=[AGENT_REASONING]),                # Layer 5: LLM verbalization
            ]
            plan.dependencies = {
                AGENT_METADATA: [],
                AGENT_VECTOR: [],
                AGENT_EVIDENCE: [AGENT_METADATA, AGENT_VECTOR],
                AGENT_FUSION: [AGENT_EVIDENCE],
                AGENT_VERIFICATION: [AGENT_FUSION],
                AGENT_REASONING: [AGENT_VERIFICATION],
            }
            plan.estimated_tokens = 200
            plan.estimated_latency_ms = 500
            plan.estimated_tools = 2
            plan.estimated_llm_calls = 1
        elif intent == "behavioral_investigation" or getattr(query_intent, 'operation', '') == "behavioral_investigation" or "behavior_analysis" in getattr(query_intent, 'required_capabilities', []):
            # Behavioral/event: ALL retrieval → normalization → fusion → verification → reasoning
            plan.agents = [AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, AGENT_EVENT, AGENT_EVIDENCE, AGENT_FUSION, AGENT_VERIFICATION, AGENT_REASONING]
            plan.tools = [
                ToolRequirement(name="postgres", required=True),
                ToolRequirement(name="milvus", required=True),
                ToolRequirement(name="s3", required=False)
            ]
            plan.execution_groups = [
                ExecutionGroup(agents=[AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, AGENT_EVENT]),  # Layer 1: ALL retrieval
                ExecutionGroup(agents=[AGENT_EVIDENCE]),                 # Layer 2: normalization
                ExecutionGroup(agents=[AGENT_FUSION]),                   # Layer 3: fusion + dedup
                ExecutionGroup(agents=[AGENT_VERIFICATION]),             # Layer 4: verification → contract
                ExecutionGroup(agents=[AGENT_REASONING]),                # Layer 5: LLM verbalization
            ]
            plan.dependencies = {
                AGENT_METADATA: [],
                AGENT_VECTOR: [],
                AGENT_VIDEO: [],
                AGENT_EVENT: [],
                AGENT_EVIDENCE: [AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, AGENT_EVENT],
                AGENT_FUSION: [AGENT_EVIDENCE],
                AGENT_VERIFICATION: [AGENT_FUSION],
                AGENT_REASONING: [AGENT_VERIFICATION],
            }
            plan.estimated_tokens = 1500
            plan.estimated_latency_ms = 1200
            plan.estimated_tools = 3
            plan.estimated_llm_calls = 2
        elif intent == "report":
            plan.agents = [AGENT_METADATA, AGENT_REPORT]
            plan.execution_groups = [ExecutionGroup(agents=[AGENT_METADATA]), ExecutionGroup(agents=[AGENT_REPORT])]
            plan.dependencies = {AGENT_REPORT: [AGENT_METADATA]}
            plan.estimated_tokens = 1000
            plan.estimated_latency_ms = 800
        else:
            # Default: ALL retrieval → normalization → fusion → verification → reasoning
            plan.agents = [AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, AGENT_EVENT, AGENT_EVIDENCE, AGENT_FUSION, AGENT_VERIFICATION, AGENT_REASONING]
            plan.tools = [
                ToolRequirement(name="postgres", required=True),
                ToolRequirement(name="milvus", required=True),
                ToolRequirement(name="s3", required=False)
            ]
            plan.execution_groups = [
                ExecutionGroup(agents=[AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, AGENT_EVENT]),  # Layer 1: ALL retrieval
                ExecutionGroup(agents=[AGENT_EVIDENCE]),                 # Layer 2: normalization
                ExecutionGroup(agents=[AGENT_FUSION]),                   # Layer 3: fusion + dedup
                ExecutionGroup(agents=[AGENT_VERIFICATION]),             # Layer 4: verification → contract
                ExecutionGroup(agents=[AGENT_REASONING]),                # Layer 5: LLM verbalization
            ]
            plan.dependencies = {
                AGENT_METADATA: [],
                AGENT_VECTOR: [],
                AGENT_VIDEO: [],
                AGENT_EVENT: [],
                AGENT_EVIDENCE: [AGENT_METADATA, AGENT_VECTOR, AGENT_VIDEO, AGENT_EVENT],
                AGENT_FUSION: [AGENT_EVIDENCE],
                AGENT_VERIFICATION: [AGENT_FUSION],
                AGENT_REASONING: [AGENT_VERIFICATION],
            }
            plan.requires_vlm = True
            plan.estimated_tokens = 6000
            plan.estimated_latency_ms = 2000
            plan.estimated_tools = 3
            plan.estimated_llm_calls = 2

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