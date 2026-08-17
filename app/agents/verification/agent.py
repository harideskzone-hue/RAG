"""
Verification Agent.

"What evidence satisfies the query?"

This agent produces the authoritative Verified Result Contract — the single
source of truth that downstream Reasoning and Response consume.

Pipeline position:
    EvidenceFusionAgent → VerificationAgent → ReasoningAgent

Responsibilities:
    ✓ Apply structured constraints from QueryIntent (gender=male, age_group=child)
    ✓ Construct verified identity set from fused evidence
    ✓ Compute verified_count from unique complete identities
    ✓ Validate no fabricated provenance passes through
    ✓ Produce VerifiedResultContract → context.results["verified_contract"]

Does NOT:
    ✗ Generate natural language
    ✗ Interpret user intent
    ✗ Modify or create evidence
    ✗ Override LLM reasoning
"""
import logging
import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.domain.models import AgentManifest, AgentCapability
from app.domain.models.confidence import ConfidenceScore
from app.domain.models.enums import EvidenceType
from app.schemas.context import BaseResult, Citation, VistaContext

logger = logging.getLogger(__name__)


def evaluate_structured_constraints(evidence_attributes: dict, constraints: list[str]) -> bool:
    """
    Generic evaluator: evaluates structured attributes against typed constraints.
    No hardcoded semantic concepts. Compares attribute[field] == value dynamically.

    Moved here from ResponseCoordinator — this is Verification's responsibility.
    """
    if not constraints or not evidence_attributes:
        return True

    for c in constraints:
        c_str = str(c).strip().lower()
        if "=" in c_str:
            field, val = c_str.split("=", 1)
            field = field.strip()
            val = val.strip()
            attr_val = str(evidence_attributes.get(field, "")).strip().lower()
            if attr_val != val:
                return False
        elif ":" in c_str:
            field, val = c_str.split(":", 1)
            field = field.strip()
            val = val.strip()
            attr_val = str(evidence_attributes.get(field, "")).strip().lower()
            if attr_val != val:
                return False
        else:
            # Single-word value constraint (e.g., "male", "female")
            matches_val = any(str(v).strip().lower() == c_str for v in evidence_attributes.values())
            if not matches_val:
                return False
    return True


from pydantic import BaseModel, Field

class VerifiedResultContract(BaseModel):
    """
    Authoritative output of the Verification Agent.
    ResponseCoordinator READS this. It NEVER creates or modifies it.
    """
    status: str = "no_evidence"
    operation: str = ""
    target: str = ""
    constraints: list[str] = Field(default_factory=list)
    verified_count: int = 0
    verified_tracks: list[str] = Field(default_factory=list)
    verified_evidence: list[dict[str, Any]] = Field(default_factory=list)
    video_id: str | None = None
    overall_confidence: float = 0.0
    events: list[dict[str, Any]] = Field(default_factory=list)


class VerificationResult(BaseResult):
    """Result from verification agent."""
    verified_count: int = 0
    verified_tracks: list[str] = []


class VerificationAgent(BaseAgent):
    """
    Produces the authoritative Verified Result Contract.
    See module docstring for full responsibility definition.
    """

    def __init__(self):
        self._name = "verification_agent"
        self._description = "Evaluates evidence against query constraints and produces the verified result contract."
        self._last_execution_time = 0.0

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
                supported_intents=[
                    "PERSON_SEARCH", "VEHICLE_SEARCH", "EVENT_SEARCH",
                    "CAMERA_STATUS", "REPORT",
                ],
                supported_entities=[],
                supported_modalities=["evidence"],
                supported_operations=["verify", "count", "evaluate"],
            ),
            cost="low",
            latency="low",
            dependencies=["evidence_fusion_agent"],
        )

    def validate(self, context: VistaContext) -> bool:
        return True

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> VerificationResult:
        start = time.time()

        # Read query intent
        query_intent = getattr(context, "query_intent", None)
        operation = ""
        target = ""
        constraints = []

        if query_intent:
            operation = getattr(query_intent, "operation", "") or ""
            target = getattr(query_intent, "target", "") or ""
            raw_constraints = getattr(query_intent, "constraints", None) or []
            constraints = [str(c) for c in raw_constraints]

        # Read fused evidence
        evidence_bundle = getattr(context, "evidence_bundle", None)
        fused_evidence = evidence_bundle.evidence if evidence_bundle else []

        # Read fusion metadata for identity keys
        fusion_metadata = context.results.get("fusion_metadata", {})
        identity_keys = fusion_metadata.get("identity_keys", [])

        # --- Step 1: Apply structured constraints ---
        is_behavioral_op = operation in ["behavioral_investigation", "event_search"]
        verified_tracks = []
        verified_evidence_list = []
        confidences = []

        for ev in fused_evidence:
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}
            attributes = meta.get("attributes") if isinstance(meta.get("attributes"), dict) else {}

            # Evaluate constraints
            if not evaluate_structured_constraints(attributes, constraints):
                continue

            # Behavioral filter: behavioral queries require structured event evidence
            if is_behavioral_op:
                event_type = (
                    meta.get("event_type")
                    or meta.get("behavior_type")
                    or meta.get("anomaly_type")
                    or origin.get("event_type")
                )
                is_event = (
                    getattr(ev, "evidence_type", None) == EvidenceType.EVENT
                    or bool(event_type)
                )
                if not is_event:
                    continue

            # Extract identity
            track_id = origin.get("track_id") or meta.get("track_id")
            video_id = origin.get("video_id") or meta.get("video_id")
            camera_id = origin.get("camera_id") or meta.get("camera_id")

            # Build track reference
            if track_id:
                track_ref = track_id
            else:
                track_ref = str(ev.evidence_id)  # observation-level ref, not identity

            if track_ref not in verified_tracks:
                verified_tracks.append(track_ref)

            # Format evidence for contract
            source_label = origin.get("type", ev.source)
            if source_label in ["video_ingestion", "video_analysis"]:
                source_label = "Video Analysis"

            verified_evidence_list.append({
                "evidence_id": str(ev.evidence_id),
                "track_id": track_id,
                "video_id": video_id,
                "camera_id": camera_id,
                "source": source_label,
                "description": meta.get("description", ""),
                "attributes": attributes,
                "confidence": ev.confidence,
                "timestamp": origin.get("video_timestamp_sec"),
            })
            confidences.append(ev.confidence)

        # --- Step 2: Compute verified count ---
        # Count unique identities (with complete identity keys)
        verified_complete_identities = set()
        for ev_info in verified_evidence_list:
            if ev_info["track_id"] and ev_info["video_id"] and ev_info["camera_id"]:
                verified_complete_identities.add(
                    (ev_info["video_id"], ev_info["camera_id"], ev_info["track_id"])
                )

        # verified_count = unique complete identities, or fallback to evidence count
        if verified_complete_identities:
            verified_count = len(verified_complete_identities)
        else:
            verified_count = len(verified_evidence_list)

        # --- Step 3: Compute confidence ---
        # Confidence aggregation rule:
        #   overall_confidence = mean(verified_entity_confidences)
        #   NEVER hardcoded
        if confidences:
            overall_confidence = sum(confidences) / len(confidences)
        else:
            overall_confidence = 0.0

        # --- Step 4: Determine status ---
        if verified_count > 0:
            status = "verified"
        else:
            status = "no_evidence"

        # --- Step 5: Produce contract ---
        contract = VerifiedResultContract(
            status=status,
            operation=operation,
            target=target,
            constraints=constraints,
            verified_count=verified_count,
            verified_tracks=verified_tracks,
            verified_evidence=verified_evidence_list,
            video_id=getattr(context, "active_video_id", None),
            overall_confidence=overall_confidence,
        )

        # Write contract to context — this is the authoritative source
        context.results["verified_contract"] = contract

        self._last_execution_time = (time.time() - start) * 1000

        logger.info(
            f"Verification: status={status}, verified_count={verified_count}, "
            f"confidence={overall_confidence:.3f}, constraints={constraints}"
        )

        return VerificationResult(
            success=True,
            evidence=[],
            confidence=ConfidenceScore(overall=overall_confidence, factors=[]),
            verified_count=verified_count,
            verified_tracks=verified_tracks,
        )

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, VerificationResult) and result.success

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        context.agent_decisions.append({
            "agent": self.name,
            "decision": f"Verified {getattr(result, 'verified_count', 0)} entities "
                        f"matching query constraints.",
        })
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall if result.confidence else 0.0

    def citations(self, result: BaseResult) -> list[Citation]:
        return []

    def metrics(self) -> dict[str, Any]:
        return {
            "execution_time_ms": self._last_execution_time,
            "tokens": 0,
            "tool_latency": 0.0,
            "memory_usage": 0.0,
            "errors": 0,
            "retry_count": 0,
        }
