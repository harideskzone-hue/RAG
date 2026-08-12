import asyncio
from typing import Any

from app.graph.supervisor.cancellation_manager import CancellationManager
from app.graph.supervisor.dispatcher import Dispatcher
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.failure_handler import FailureHandler
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.graph.supervisor.result_collector import ResultCollector
from app.graph.supervisor.retry_manager import RetryManager
from app.graph.supervisor.scheduler import Scheduler
from app.graph.supervisor.timeout_manager import TimeoutManager
from app.schemas.context import VistaContext
from app.domain.models.enums import ExecutionMode
from app.domain.confidence.aggregator import ConfidenceAggregator
from app.graph.supervisor.state_machine import StateMachine, ExecutionState


class Supervisor:
    """
    The Orchestration Engine.
    Externally provides a clean .run(context) API.
    Internally delegates to single-responsibility modules.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.state_machine = StateMachine()
        self.event_bus = EventBus()
        self.retry_manager = RetryManager()
        self.failure_handler = FailureHandler()
        self.result_collector = ResultCollector()
        self.dispatcher = Dispatcher()
        self.cancellation_manager = CancellationManager(self.state_machine)
        self.timeout_manager = TimeoutManager()
        
        self.scheduler = Scheduler(
            dispatcher=self.dispatcher,
            retry_manager=self.retry_manager,
            failure_handler=self.failure_handler,
            result_collector=self.result_collector,
            state_machine=self.state_machine,
            event_bus=self.event_bus
        )
        
        self.response_coordinator = ResponseCoordinator()

    async def run(self, context: VistaContext) -> dict[str, Any]:
        """
        Main entry point for executing a validated plan.
        Auto-generates an execution plan if not provided.
        """
        if context.execution_plan is None:
            from app.agents.intent.classifier import HybridIntentClassifier
            from app.agents.planner.planner import ExecutionPlanner
            
            classifier = HybridIntentClassifier(llm_client=self.llm_client)
            intent_result = await classifier.classify(context.current_query)
            context.results["intent_agent"] = intent_result
            if hasattr(intent_result, 'query_intent'):
                context.query_intent = intent_result.query_intent
            planner = ExecutionPlanner(llm_client=self.llm_client)
            context.execution_plan = await planner.plan(intent_result, context.current_query)

        self.state_machine.transition_to(ExecutionState.RUNNING)
        
        try:
            max_replans = 3
            replans = 0
            
            # Simple Mode: run once
            if context.execution_mode == ExecutionMode.SIMPLE:
                await self.scheduler.execute_groups(context)
                self.state_machine.transition_to(ExecutionState.COMPLETED)
                await self._persist_state(context)
                return self.response_coordinator.generate_response(context)
                
            # Iterative / Investigation Mode
            while True:
                # Persist state at start of iteration
                await self._persist_state(context)
                
                # Execute current groups/tasks
                await self.scheduler.execute_groups(context)
                
                # Ask Policy Engine if we should continue (budget, confidence, etc.)
                from app.domain.policy.engine import PolicyEngine
                from app.domain.policy.repository import InMemoryPolicyRepository
                from app.domain.policy.context import PolicyContext
                from app.domain.policy.decision import PolicyDecision
                
                policy_engine = PolicyEngine(InMemoryPolicyRepository())
                policy_context = PolicyContext(current_cost_usd=0.0, current_latency_ms=0.0, iteration_count=replans)
                explanation, trace = policy_engine.evaluate_plan(policy_context, context.execution_plan)
                
                if explanation.decision == PolicyDecision.REJECT:
                    break
                
                # Check for new actions requested by the Reasoning Engine
                new_agents = []
                for result in context.results.values():
                    if hasattr(result, "metadata") and "next_actions" in result.metadata:
                        for action in result.metadata["next_actions"]:
                            agent_name = action.get("agent")
                            if agent_name and agent_name not in context.execution_plan.agents:
                                new_agents.append(agent_name)
                                
                if not new_agents:
                    # No new evidence/actions requested, stop iterating
                    break
                    
                if replans >= max_replans:
                    break
                    
                replans += 1
                
                # The Policy Engine is responsible for modifying the plan, we just apply it
                if explanation.validated_plan:
                    context.execution_plan = explanation.validated_plan
                else:
                    context.execution_plan.agents.extend(new_agents)
                    context.execution_plan.execution_groups.append(new_agents)
                    
                # Clear out next_actions so we don't loop forever
                for result in context.results.values():
                    if hasattr(result, "metadata") and "next_actions" in result.metadata:
                        result.metadata["next_actions"] = []
            
            if self.state_machine.state not in [ExecutionState.FAILED, ExecutionState.CANCELLED]:
                self.state_machine.transition_to(ExecutionState.COMPLETED)
                
            # Compute final metrics
            context.metrics.iterations = replans + 1
            if context.execution_ledger:
                context.metrics.total_latency_ms = sum(record.execution_time_ms for record in context.execution_ledger)
                for record in context.execution_ledger:
                    context.metrics.agent_utilization[record.agent_name] = context.metrics.agent_utilization.get(record.agent_name, 0) + 1
                successes = sum(1 for r in context.execution_ledger if r.status == ExecutionState.COMPLETED)
                context.metrics.success_rate = successes / len(context.execution_ledger)
            
            await self._persist_state(context)
            return self.response_coordinator.generate_response(context)
            
        except asyncio.CancelledError:
            self.cancellation_manager.cancel()
            await self._persist_state(context)
        except Exception as e:
            # Ultimate failure catcher
            self.state_machine.transition_to(ExecutionState.FAILED)
            context.results["error"] = str(e)
            await self._persist_state(context)
            
        return self.response_coordinator.generate_response(context)

    async def _persist_state(self, context: VistaContext):
        """Persists the supervisor context state to CheckpointStore."""
        try:
            from app.graph.supervisor.store import get_checkpoint_store
            store = get_checkpoint_store()
            await store.save(context)
        except Exception as e:
            # Non-blocking failure for persistence
            import logging
            logging.getLogger(__name__).error(f"Failed to persist state: {e}")
