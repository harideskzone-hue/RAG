import asyncio
import time

from app.graph.supervisor.dispatcher import Dispatcher
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.failure_handler import FailureHandler
from app.graph.supervisor.result_collector import ResultCollector
from app.graph.supervisor.retry_manager import RetryManager
from app.graph.supervisor.state_machine import ExecutionState, StateMachine
from app.graph.supervisor.telemetry import AgentEvent
from app.schemas.context import VistaContext, AgentExecutionRecord


from app.graph.supervisor.timeout_manager import TimeoutManager

class Scheduler:
    """
    Executes agents based on the execution_groups defined in the plan.
    Handles parallel execution inside groups, and sequential execution between groups.
    """
    def __init__(self, dispatcher: Dispatcher, retry_manager: RetryManager, 
                 failure_handler: FailureHandler, result_collector: ResultCollector,
                 state_machine: StateMachine, event_bus: EventBus,
                 timeout_manager: TimeoutManager = None):
        self.dispatcher = dispatcher
        self.retry_manager = retry_manager
        self.failure_handler = failure_handler
        self.result_collector = result_collector
        self.state_machine = state_machine
        self.event_bus = event_bus
        self.timeout_manager = timeout_manager or TimeoutManager()

    async def execute_groups(self, context: VistaContext):
        groups = context.execution_plan.execution_groups
        
        for group_idx, group in enumerate(groups):
            if self.state_machine.state in [ExecutionState.FAILED, ExecutionState.CANCELLED]:
                break
                
            # Execute agents in the current group in parallel
            tasks = [self._execute_agent_with_retries(agent_name, context) for agent_name in group]
            await asyncio.gather(*tasks)

    async def _execute_agent_with_retries(self, agent_name: str, context: VistaContext):
        start_time = time.time()
        
        # Publish start event
        self.event_bus.publish(AgentEvent(
            agent_name=agent_name,
            event_type="START",
            start_time=start_time,
            status="RUNNING",
            trace_id=context.conversation_id
        ))
        
        while True:
            if self.state_machine.state == ExecutionState.CANCELLED:
                break
                
            try:
                coro = self.dispatcher.dispatch(agent_name, context)
                result = await self.timeout_manager.execute_with_timeout(coro)
                self.result_collector.collect(agent_name, result, context)
                
                # Append to Ledger
                latency = (time.time() - start_time) * 1000
                context.execution_ledger.append(AgentExecutionRecord(
                    agent_name=agent_name,
                    task_id=f"{agent_name}_{int(start_time)}",
                    output=result.model_dump() if hasattr(result, "model_dump") else {},
                    status="completed",
                    execution_time_ms=latency,
                    confidence=getattr(result.confidence, "overall", result.confidence) if hasattr(result, "confidence") else 0.0,
                    retry_count=self.retry_manager.retry_counts.get(agent_name, 0),
                    timestamp=time.time()
                ))
                
                # Publish complete event
                self.event_bus.publish(AgentEvent(
                    agent_name=agent_name,
                    event_type="COMPLETE",
                    start_time=start_time,
                    end_time=time.time(),
                    status="SUCCESS",
                    latency_ms=(time.time() - start_time) * 1000,
                    trace_id=context.conversation_id
                ))
                break
                
            except asyncio.CancelledError:
                self.state_machine.transition_to(ExecutionState.CANCELLED)
                raise
            except Exception as e:
                # Publish error event
                self.event_bus.publish(AgentEvent(
                    agent_name=agent_name,
                    event_type="ERROR",
                    start_time=start_time,
                    end_time=time.time(),
                    status="ERROR",
                    errors=[str(e)],
                    trace_id=context.conversation_id
                ))
                
                context.execution_ledger.append(AgentExecutionRecord(
                    agent_name=agent_name,
                    task_id=f"{agent_name}_{int(start_time)}",
                    status="failed",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    confidence=0.0,
                    retry_count=self.retry_manager.retry_counts.get(agent_name, 0),
                    error=str(e),
                    timestamp=time.time()
                ))
                
                if self.retry_manager.should_retry(agent_name, e):
                    if self.state_machine.state in [ExecutionState.FAILED, ExecutionState.CANCELLED]:
                        break
                    self.state_machine.transition_to(ExecutionState.RETRYING)
                    backoff_delay = self.retry_manager.get_backoff_delay(agent_name)
                    await asyncio.sleep(backoff_delay)
                    if self.state_machine.state in [ExecutionState.FAILED, ExecutionState.CANCELLED]:
                        break
                    self.state_machine.transition_to(ExecutionState.RUNNING)
                    continue
                
                # If we exhaust retries, handle ultimate failure
                can_continue = self.failure_handler.handle_failure(agent_name, e, context, self.state_machine)
                if not can_continue:
                    break # Fatal error, abort
                else:
                    break # Graceful degradation, skip this agent but continue workflow
