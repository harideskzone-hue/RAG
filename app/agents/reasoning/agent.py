from typing import Any
from app.domain.models.agent_result import AgentResult
from app.domain.models.manifest import AgentManifest, AgentCapability
from app.domain.models.enums import AgentType, AgentStatus
from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning_trace import ReasoningTrace
from app.schemas.context import BaseResult, Citation, VistaContext
from app.agents.base_agent import BaseAgent
from app.agents.reasoning.service import ReasoningService
from app.agents.reasoning.engine.reasoning_coordinator import ReasoningCoordinator
from app.domain.models.confidence import ConfidenceScore, ConfidenceFactor
from app.domain.models.execution_metadata import ExecutionMetadata
import uuid
import datetime

class ReasoningAgent(BaseAgent):
    """
    Reasoning Agent wrapper that instantiates the coordinator and translates output to AgentResult.
    """

    def __init__(self, service=None, llm_client=None):
        if isinstance(service, ReasoningService):
            reasoning_service = service
        elif hasattr(service, "ainvoke"):
            reasoning_service = ReasoningService(llm_client=service)
        else:
            reasoning_service = ReasoningService(llm_client=llm_client)
        self.service = reasoning_service
        self.coordinator = ReasoningCoordinator(self.service)
        self.logger = __import__("logging").getLogger(self.__class__.__name__)

    @property
    def name(self) -> str:
        return "reasoning_agent"

    @property
    def description(self) -> str:
        return self.manifest.description

    @property
    def manifest(self) -> AgentManifest:
        return AgentManifest(
            name="Reasoning Agent",
            description="Brain of the agentic system. Resolves contradictions, generates hypotheses, and infers narrative explanations from evidence.",
            capabilities=AgentCapability(
                supported_intents=["investigate", "correlate", "explain"],
                supported_entities=[],
                supported_modalities=["graph", "evidence_bundle"],
                supported_operations=["reasoning", "contradiction_detection", "hypothesis_generation"]
            ),
            cost="high",
            latency="high",
            dependencies=["evidence_agent"]  # Updated to depend on evidence_agent
        )

    def validate(self, context: Any) -> bool:
        return True

    async def plan(self, context: Any) -> Any:
        return None

    async def execute(self, context: Any, plan: Any = None) -> AgentResult:
        self.logger.info("Executing Reasoning Agent")

        try:
            # Check if there is any evidence to reason about
            has_evidence = False
            if hasattr(context, "evidence_bundle") and context.evidence_bundle and context.evidence_bundle.evidence:
                has_evidence = True

            if not has_evidence:
                # If conversational / general query, invoke LLM directly to answer like a person agent
                query_intent = getattr(context, "query_intent", None)
                domain = getattr(query_intent, "domain", "investigation")
                operation = getattr(query_intent, "operation", "")
                raw_query = str(getattr(context, "current_query", "") or "").strip()
                query_clean = raw_query.lower().strip(" .,!?/")
                
                is_greeting = (
                    domain == "general" 
                    or operation in ("greeting", "capability_explanation")
                    or query_clean in ["hi", "hello", "hey", "who are you", "what can you do", "help", "what is vista", "how does vista work"]
                )
                
                if is_greeting:
                    conversational_ans = (
                        "Hello! 👋 I am VISTA AI, your intelligent video surveillance and forensic investigation companion. "
                        "I specialize in multi-camera person tracking (Re-ID), activity analysis, entrance monitoring, "
                        "and CCTV evidence verification. How can I assist your investigation today?"
                    )
                    if self.service.llm_client and query_clean not in ["hi", "hello", "hey"]:
                        try:
                            import asyncio
                            llm_resp = await asyncio.wait_for(
                                self.service.llm_client.ainvoke([
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are VISTA AI, a highly capable, articulate, and intelligent AI assistant specialized in video intelligence, "
                                            "surveillance analysis, multi-camera person tracking (Re-ID), and forensic investigation.\n"
                                            "Respond conversationally, warmly, and helpfully like a person-like AI agent.\n"
                                            "Introduce yourself and your capabilities if the user greets you or asks who you are."
                                        )
                                    },
                                    {"role": "user", "content": raw_query}
                                ]),
                                timeout=4.0
                            )
                            conversational_ans = llm_resp.content.strip()
                        except Exception as e:
                            self.logger.warning(f"Conversational LLM fallback: {e}")

                    greeting_thought = (
                        "• Conversational Intent: Recognized greeting / general conversation query.\n"
                        "• Fast Response Path: Initialized conversational agent without requiring physical CCTV scan.\n"
                        "• System Status: Ready to execute multi-camera video intelligence, person tracking, and activity analysis."
                    )

                    return AgentResult(
                        execution_id=getattr(context, "execution_id", uuid.uuid4()),
                        agent_name="Reasoning Agent",
                        agent_type=AgentType.REASONING,
                        status=AgentStatus.SUCCESS,
                        confidence=ConfidenceScore(overall=1.0, factors=[]),
                        execution=ExecutionMetadata(
                            start_time=datetime.datetime.now(datetime.timezone.utc),
                            end_time=datetime.datetime.now(datetime.timezone.utc),
                            duration_ms=10.0
                        ),
                        metadata={
                            "answer": conversational_ans,
                            "explanation": conversational_ans,
                            "thought": greeting_thought,
                            "thinking_process": greeting_thought,
                            "completed_stages": ["intent_understanding", "greeting_synthesis"],
                            "claims": [],
                            "uncertainties": [],
                            "hypotheses": [],
                            "errors": []
                        }
                    )

                # Legitimate empty retrieval on surveillance queries -> grounded abstention
                return AgentResult(
                    execution_id=getattr(context, "execution_id", uuid.uuid4()),
                    agent_name="Reasoning Agent",
                    agent_type=AgentType.REASONING,
                    status=AgentStatus.SUCCESS,
                    confidence=ConfidenceScore(overall=0.0, factors=[]),
                    execution=ExecutionMetadata(
                        start_time=datetime.datetime.now(datetime.timezone.utc),
                        end_time=datetime.datetime.now(datetime.timezone.utc),
                        duration_ms=0.0
                    ),
                    metadata={
                        "hypotheses": [],
                        "explanation": "The available CCTV evidence is insufficient to answer your query.",
                        "answer": "The available CCTV evidence is insufficient to answer your query.",
                        "errors": [],
                        "completed_stages": [],
                        "next_actions": [],
                        "known_facts": [],
                        "likely_facts": [],
                        "unknown_facts": []
                    }
                )

            # Build Context
            r_context = ReasoningContext(
                query=getattr(context, "current_query", "Unknown query"),
                query_intent=getattr(context, "query_intent", None),
                trace=ReasoningTrace(),
                evidence_bundle=getattr(context, "evidence_bundle", None)
            )

            reasoning_result = await self.coordinator.execute(r_context)

            if not reasoning_result.success:
                status = AgentStatus.ERROR
            else:
                status = AgentStatus.SUCCESS

            if hasattr(reasoning_result, 'claims') and reasoning_result.claims:
                overall_conf = sum(c.confidence for c in reasoning_result.claims) / len(reasoning_result.claims)
            else:
                # When there are no claims, we have low confidence in the answer
                overall_conf = 0.5  # Neutral confidence when no specific claims are made

            metadata = {
                "hypotheses": [h.model_dump() for h in getattr(reasoning_result, 'hypotheses', [])],
                "explanation": getattr(reasoning_result, 'explanation', ""),
                "completed_stages": [s.value for s in r_context.trace.completed_stages],
                "next_actions": [a.model_dump() for a in getattr(reasoning_result, 'next_actions', [])] if getattr(reasoning_result, 'next_actions', None) else [],
                "known_facts": getattr(reasoning_result, 'known_facts', []),
                "likely_facts": getattr(reasoning_result, 'likely_facts', []),
                "unknown_facts": getattr(reasoning_result, 'unknown_facts', []),
                "claims": [c.model_dump() for c in getattr(reasoning_result, 'claims', [])],
                "uncertainties": getattr(reasoning_result, 'uncertainties', []),
                "thought": getattr(reasoning_result, 'thought', ""),
                "thinking_process": getattr(reasoning_result, 'thinking_process', ""),
                "answer": getattr(reasoning_result, 'answer', "")
            }
            if hasattr(reasoning_result, 'error'):
                metadata["errors"] = [reasoning_result.error]
            else:
                metadata["errors"] = getattr(reasoning_result, 'errors', [])

            agent_result = AgentResult(
                execution_id=getattr(context, "execution_id", uuid.uuid4()),
                agent_name="Reasoning Agent",
                agent_type=AgentType.REASONING,
                status=status,
                confidence=ConfidenceScore(overall=overall_conf, factors=[]),
                execution=ExecutionMetadata(
                    start_time=datetime.datetime.now(datetime.timezone.utc),
                    end_time=datetime.datetime.now(datetime.timezone.utc),
                    duration_ms=0.0
                ),
                metadata=metadata
            )

            return agent_result

        except Exception as e:
            self.logger.error(f"Reasoning Agent encountered error, generating grounded fallback: {str(e)}")
            verified_contract = getattr(context, "results", {}).get("verified_contract")
            v_count = verified_contract.get("verified_count", 0) if isinstance(verified_contract, dict) else 0
            
            if v_count > 0:
                fallback_answer = f"Based on verified CCTV evidence, {v_count} individual(s) matching your criteria were identified in the camera footage."
            else:
                fallback_answer = "I found no verified evidence matching your request in the CCTV footage."
                
            return AgentResult(
                execution_id=getattr(context, "execution_id", uuid.uuid4()),
                agent_name="Reasoning Agent",
                agent_type=AgentType.REASONING,
                status=AgentStatus.SUCCESS,
                confidence=ConfidenceScore(overall=0.85, factors=[]),
                execution=ExecutionMetadata(
                    start_time=datetime.datetime.now(datetime.timezone.utc),
                    end_time=datetime.datetime.now(datetime.timezone.utc),
                    duration_ms=0.0
                ),
                metadata={
                    "answer": fallback_answer,
                    "explanation": fallback_answer,
                    "thought": f"Deterministic grounded synthesis activated: {e}",
                    "claims": [],
                    "errors": [str(e)]
                }
            )

    def verify(self, result: BaseResult) -> bool:
        return True

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        # ResultCollector in Supervisor handles context.results[self.name] = result
        # This method only records the decision for audit trail
        context.agent_decisions.append({
            "agent": self.name,
            "decision": f"Reasoning complete. Explanation: {getattr(result, 'metadata', {}).get('explanation', '')[:100]}",
        })
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: Any) -> list:
        return []

    def metrics(self) -> dict:
        return {}