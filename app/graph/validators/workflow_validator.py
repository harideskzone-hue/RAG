from app.graph.validators.schemas import ValidationResult
from app.schemas.context import ExecutionPlan, VistaContext


class WorkflowValidator:
    """
    Acts as a compiler for the ExecutionPlan before the Supervisor receives it.
    Ensures that the plan is structurally sound, safe, and cost-effective.
    """
    
    def __init__(self, max_token_limit: int = 100000, max_latency_ms: int = 10000):
        self.max_token_limit = max_token_limit
        self.max_latency_ms = max_latency_ms
        # Define RBAC rules (simplified for demonstration)
        self.role_permissions = {
            "operator": {"agents": ["metadata", "vector", "video", "report"], "tools": ["postgres", "milvus", "s3"]},
            "admin": {"agents": ["metadata", "vector", "video", "report", "event"], "tools": ["postgres", "milvus", "s3", "websocket", "delete_tool"]}
        }

    def validate(self, context: VistaContext) -> ValidationResult:
        result = ValidationResult(valid=True)
        plan: ExecutionPlan = context.execution_plan
        
        if not plan:
            result.valid = False
            result.errors.append("No ExecutionPlan found in context.")
            return result

        self._validate_confidence(context, result)
        self._validate_agents(plan, result)
        self._validate_tools(plan, result)
        self._validate_dependencies_and_groups(plan, result)
        self._validate_permissions(context, plan, result)
        self._validate_cost_and_latency(plan, result)
        self._validate_risk(plan, result)

        if len(result.errors) > 0:
            result.valid = False
            
        return result

    def _validate_confidence(self, context: VistaContext, result: ValidationResult):
        # We can simulate confidence checking based on the intent result confidence
        # or if the planner itself assigned a confidence.
        # Let's say intent agent confidence is in context.results["intent_agent"]
        if "intent_agent" in context.results:
            intent_conf = context.results["intent_agent"].confidence
            if intent_conf < 0.25:
                result.errors.append(f"Confidence too low ({intent_conf}). Requires clarification.")

    def _validate_agents(self, plan: ExecutionPlan, result: ValidationResult):
        if not plan.agents:
            result.errors.append("No agents specified in ExecutionPlan.")
            return

        unique_agents = set()
        for agent_name in plan.agents:
            if agent_name in unique_agents:
                result.errors.append(f"Duplicate agent found: {agent_name}")
            unique_agents.add(agent_name)
            
            # Since we are building incrementally, we don't strictly enforce agent_registry existence 
            # for tests yet unless we register them. We will just check the structure.
            # In production:
            # if agent_name not in agent_registry.get_all_agents():
            #     result.errors.append(f"Unknown agent: {agent_name}")

    def _validate_tools(self, plan: ExecutionPlan, result: ValidationResult):
        # Tools depend on specific agents. E.g., milvus needs vector agent.
        tool_names = [t.name for t in plan.tools]
        
        if "milvus" in tool_names and "vector" not in plan.agents:
            result.errors.append("Tool 'milvus' requires the 'vector' agent.")
            
        if "s3" in tool_names and "video" not in plan.agents:
            result.errors.append("Tool 's3' requires the 'video' agent.")
            
        # Tool registry check (mocked for now, assumes registered if in tests or we register them later)

    def _validate_dependencies_and_groups(self, plan: ExecutionPlan, result: ValidationResult):
        # Check for circular dependencies
        graph = plan.dependencies
        visited = set()
        path = set()

        def visit(node):
            if node in path:
                return False # Cycle detected
            if node in visited:
                return True
            
            path.add(node)
            for neighbor in graph.get(node, []):
                if not visit(neighbor):
                    return False
            path.remove(node)
            visited.add(node)
            return True

        for node in graph:
            if not visit(node):
                result.errors.append("Circular dependency detected in execution plan.")
                return # Stop further dependency validation

        # Check execution groups against dependencies
        # An agent cannot be in the same group or an earlier group than its dependencies
        agent_to_group_idx = {}
        for idx, group in enumerate(plan.execution_groups):
            for agent in group:
                agent_to_group_idx[agent] = idx
                
        for agent, deps in plan.dependencies.items():
            agent_idx = agent_to_group_idx.get(agent, -1)
            for dep in deps:
                dep_idx = agent_to_group_idx.get(dep, -1)
                if agent_idx != -1 and dep_idx != -1:
                    if agent_idx <= dep_idx:
                        result.errors.append(f"Agent '{agent}' (Group {agent_idx}) depends on '{dep}' (Group {dep_idx}), which is in the same or a later execution group.")

        # Pipeline Ordering Contract: No evidence-producing agent may execute after Verification
        evidence_agents = {"video_agent", "metadata_agent", "vector_agent", "event_agent", "evidence_fusion_agent"}
        verification_idx = agent_to_group_idx.get("verification_agent", -1)
        
        if verification_idx != -1:
            for agent, idx in agent_to_group_idx.items():
                if agent in evidence_agents and idx > verification_idx:
                    result.errors.append(
                        f"Pipeline ordering contract violated: Evidence producer '{agent}' "
                        f"(Group {idx}) scheduled after verification_agent (Group {verification_idx})."
                    )

    def _validate_permissions(self, context: VistaContext, plan: ExecutionPlan, result: ValidationResult):
        role = context.user.role.lower()
        allowed = self.role_permissions.get(role, {"agents": [], "tools": []})
        
        for agent in plan.agents:
            if agent not in allowed["agents"]:
                result.errors.append(f"Role '{role}' is not allowed to use agent '{agent}'.")
                
        for tool in plan.tools:
            if tool.name not in allowed["tools"]:
                result.errors.append(f"Role '{role}' is not allowed to use tool '{tool.name}'.")

    def _validate_cost_and_latency(self, plan: ExecutionPlan, result: ValidationResult):
        if plan.estimated_tokens > self.max_token_limit:
            result.errors.append(f"Estimated tokens ({plan.estimated_tokens}) exceeds limit ({self.max_token_limit}).")
            
        if plan.estimated_latency_ms > self.max_latency_ms:
            result.errors.append(f"Estimated latency ({plan.estimated_latency_ms}ms) exceeds SLA ({self.max_latency_ms}ms).")

    def _validate_risk(self, plan: ExecutionPlan, result: ValidationResult):
        if plan.risk_level in ["HIGH", "CRITICAL"]:
            result.approval_required = True
            result.warnings.append(f"Risk level is {plan.risk_level}. Human approval required.")
