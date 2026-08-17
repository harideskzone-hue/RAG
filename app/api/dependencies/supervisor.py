import logging
import os

from fastapi import Depends

from app.agents.confidence.agent import ConfidenceAgent
from app.agents.confidence.engine import ConfidenceEngine
from app.agents.event.agent import EventAgent
from app.agents.evidence.agent import EvidenceAgent

# Agents
from app.agents.metadata.agent import MetadataAgent
from app.agents.registry import agent_registry
from app.agents.report.agent import ReportAgent
from app.agents.vector.agent import VectorAgent
from app.agents.video.agent import VideoAgent
from app.agents.knowledge_graph.agent import KnowledgeGraphAgent
from app.agents.reasoning.agent import ReasoningAgent
from app.agents.reasoning.service import ReasoningService
from app.agents.guardrail.agent import GuardrailAgent

# Tools
from app.api.dependencies.repositories import get_vector_tool, get_postgres_tool
from app.api.dependencies.services import (
    get_event_bus,
    get_event_service,
    get_metadata_service,
    get_report_service,
    get_s3_tool,
    get_vector_service,
    get_video_service,
)
from app.domain.models.confidence import ConfidencePolicy
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.supervisor import Supervisor
from app.tools.registry import tool_registry

logger = logging.getLogger(__name__)

# Track whether registries have been initialized
_initialized = False


def _initialize_registries(
    event_bus, postgres_tool, vector_tool, s3_tool,
    metadata_service, vector_service, video_service,
    event_service, report_service
):
    """Initialize tool and agent registries once at first request."""
    global _initialized
    if _initialized and len(agent_registry.get_all_agents()) > 0:
        return
    
    # Register Tools (only if not already registered)
    if not tool_registry.get_tool(postgres_tool.name):
        tool_registry.register(postgres_tool.name, postgres_tool.execute, postgres_tool.description)
    if not tool_registry.get_tool(vector_tool.name):
        tool_registry.register(vector_tool.name, vector_tool.execute, vector_tool.description)
    if not tool_registry.get_tool(s3_tool.name):
        tool_registry.register(s3_tool.name, s3_tool.execute, s3_tool.description)
        
    # Register Agents (only if not already registered)
    confidence_engine = ConfidenceEngine(ConfidencePolicy())
    
    vlm_provider = os.environ.get("VLM_PROVIDER", "gemini")
    reasoning_provider = os.environ.get("REASONING_PROVIDER", "ollama")
    
    # Use the LLM abstraction layer — agents never import provider-specific clients
    from app.infrastructure.llm.model_registry import ModelRegistry

    try:
        reasoning_llm_client = ModelRegistry.get_client(role="reasoning")
    except Exception as e:
        logger.warning(f"LLM client initialization failed: {e}. Reasoning disabled.")
        reasoning_llm_client = ModelRegistry.get_client(provider="disabled")

    
    from app.agents.time.agent import TimeAgent
    from app.agents.evidence_fusion.agent import EvidenceFusionAgent
    from app.agents.verification.agent import VerificationAgent
    agents_to_register = [
        MetadataAgent(metadata_service),
        VectorAgent(vector_service),
        EvidenceAgent(),
        ConfidenceAgent(confidence_engine),
        VideoAgent(video_service),
        EventAgent(event_service),
        ReportAgent(report_service),
        KnowledgeGraphAgent(),
        EvidenceFusionAgent(),
        VerificationAgent(),
        ReasoningAgent(ReasoningService(reasoning_llm_client)),
        GuardrailAgent(),
        TimeAgent()
    ]
    
    for agent in agents_to_register:
        if not agent_registry.get_agent(agent.name):
            agent_registry.register(agent)
    
    _initialized = True
    logger.info("Tool and agent registries initialized.")


from app.api.dependencies.repositories import _is_depends

def get_supervisor(
    event_bus: EventBus = Depends(get_event_bus),
    postgres_tool = Depends(get_postgres_tool),
    vector_tool = Depends(get_vector_tool),
    s3_tool = Depends(get_s3_tool),
    metadata_service = Depends(get_metadata_service),
    vector_service = Depends(get_vector_service),
    video_service = Depends(get_video_service),
    event_service = Depends(get_event_service),
    report_service = Depends(get_report_service)
) -> Supervisor:
    if _is_depends(event_bus):
        event_bus = get_event_bus()
    if _is_depends(postgres_tool):
        postgres_tool = get_postgres_tool(event_bus)
    if _is_depends(vector_tool):
        vector_tool = get_vector_tool(event_bus)
    if _is_depends(s3_tool):
        s3_tool = get_s3_tool(event_bus)
    if _is_depends(metadata_service):
        metadata_service = get_metadata_service(event_bus=event_bus)
    if _is_depends(vector_service):
        vector_service = get_vector_service(event_bus=event_bus)
    if _is_depends(video_service):
        video_service = get_video_service(s3_tool=s3_tool, event_bus=event_bus)
    if _is_depends(event_service):
        event_service = get_event_service(event_bus=event_bus)
    if _is_depends(report_service):
        report_service = get_report_service(event_bus=event_bus)
    
    # Initialize registries on first call
    _initialize_registries(
        event_bus, postgres_tool, vector_tool, s3_tool,
        metadata_service, vector_service, video_service,
        event_service, report_service
    )
    
    llm_client = None
    vlm_provider = os.getenv("VLM_PROVIDER", "none").lower()
    if vlm_provider == "smolvlm256":
        try:
            from app.domain.vlm.smolvlm256_client import SmolVLM256Client
            llm_client = SmolVLM256Client()
        except Exception as e:
            logger.warning(f"Could not initialize VLM client (smolvlm256): {e}")
    elif vlm_provider == "gemini":
        try:
            from app.domain.vlm.gemini_client import GeminiVLClient
            llm_client = GeminiVLClient()
        except Exception as e:
            logger.warning(f"Could not initialize VLM client (gemini): {e}")
    elif vlm_provider == "none":
        logger.info("VLM_PROVIDER is none. Model execution is disabled for visual perception.")
    else:
        logger.warning(f"Unknown VLM_PROVIDER: {vlm_provider}")
        
    # Create Supervisor per-request (lightweight — it's just a coordinator)
    supervisor = Supervisor(llm_client=llm_client)
    supervisor.event_bus = event_bus
    return supervisor
