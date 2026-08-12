import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.intent.classifier import HybridIntentClassifier
from app.agents.intent.schemas import IntentResult
from app.schemas.context import BaseResult, Citation, VistaContext
from app.domain.models import AgentManifest, AgentCapability
from app.domain.models.enums import EntityType


class IntentAgent(BaseAgent):
    """
    Intent Classification Agent.
    Responsibilities: Classify intent + extract entities.
    Does NOT make routing decisions.
    """

    def __init__(self, llm_client=None):
        self._name = "intent_agent"
        self._description = "Classifies user queries into specific intents and extracts relevant entities."
        self.classifier = HybridIntentClassifier(llm_client)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def manifest(self) -> AgentManifest:
        return AgentManifest(
            name=self.name,
            description=self.description,
            capabilities=AgentCapability(
                supported_intents=["INTENT_CLASSIFICATION"],
                supported_entities=[],
                supported_modalities=["text"],
                supported_operations=["classify"]
            ),
            cost="low",
            latency="fast"
        )

    def validate(self, context: VistaContext) -> bool:
        # Intent agent always runs first if current_query exists and execution_plan is empty
        return bool(context.current_query) and not context.execution_plan

    async def plan(self, context: VistaContext) -> Any:
        # Intent agent doesn't need a complex plan, it just runs the classifier
        return None

    async def execute(self, context: VistaContext, plan: Any) -> IntentResult:
        start_time = time.time()
        
        # Run classification
        result = await self.classifier.classify(context.current_query)
        
        # Add basic metrics tracking to the result object dynamically (or tracked outside)
        self._last_execution_time = (time.time() - start_time) * 1000 # ms
        
        return result

    def verify(self, result: BaseResult) -> bool:
        # Verify it's an IntentResult and has an intent
        return isinstance(result, IntentResult) and result.intent is not None

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        # Populate context with results
        context.results[self.name] = result
        # Safely access IntentResult-specific fields
        if isinstance(result, IntentResult):
            context.agent_decisions.append({
                "agent": self.name,
                "decision": f"Classified intent as {result.intent.value}",
                "requires_clarification": result.requires_clarification
            })
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: BaseResult) -> list[Citation]:
        # Intent agent doesn't typically provide citations
        return []

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0),
            "tokens": 0, # To be filled if LLM was used
            "tool_latency": 0.0,
            "memory_usage": 0.0,
            "errors": 0,
            "retry_count": 0
        }
