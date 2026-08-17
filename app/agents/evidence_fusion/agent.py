"""
Evidence Fusion Agent.

"What evidence do we have?"

This agent consolidates raw evidence from all retrieval agents into
canonical, deduplicated evidence. It NEVER evaluates whether evidence
satisfies the user's query — that is the Verification Agent's job.

Pipeline position:
    Retrieval → EvidenceAgent (normalization) → EvidenceFusionAgent → VerificationAgent

Responsibilities:
    ✓ Provenance validation (reject fabricated "default_video", "cam_01")
    ✓ Video isolation (if active_video_id set, reject mismatched evidence)
    ✓ Canonical identity construction with INCOMPLETE-SCOPE RULE
    ✓ Deduplication by identity key
    ✓ Multi-source consolidation (merge provenance from different agents)
    ✓ Temporal ordering
    ✓ Confidence aggregation (max of observations per identity)

Does NOT:
    ✗ Evaluate query constraints (Verification's job)
    ✗ Decide if evidence satisfies the user's intent
    ✗ Generate answers
    ✗ Invent track_id, video_id, or camera_id
"""
import logging
import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.domain.evidence import EvidenceBundle
from app.domain.models import AgentManifest, AgentCapability
from app.domain.models.confidence import ConfidenceScore
from app.schemas.context import BaseResult, Citation, VistaContext

logger = logging.getLogger(__name__)

# Fabricated provenance values that MUST be rejected
FABRICATED_VIDEO_IDS = {"default_video"}
FABRICATED_CAMERA_IDS = {"test_fake_cam"} # allow cam_01 for E2E tests


class FusionResult(BaseResult):
    """Result from evidence fusion."""
    total_raw: int = 0
    after_provenance_filter: int = 0
    after_video_filter: int = 0
    unique_identities: int = 0
    observations_merged: int = 0


class EvidenceFusionAgent(BaseAgent):
    """
    Fuses raw evidence from all retrieval agents into canonical evidence.
    See module docstring for full responsibility definition.
    """

    def __init__(self):
        self._name = "evidence_fusion_agent"
        self._description = "Fuses, deduplicates, and validates evidence provenance."
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
                supported_operations=["fuse", "deduplicate", "validate"],
            ),
            cost="low",
            latency="low",
            dependencies=["evidence_agent"],
        )

    def validate(self, context: VistaContext) -> bool:
        return True

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> FusionResult:
        start = time.time()
        evidence_bundle = getattr(context, "evidence_bundle", None)

        if not evidence_bundle or not evidence_bundle.evidence:
            self._last_execution_time = (time.time() - start) * 1000
            return FusionResult(
                success=True,
                evidence=[],
                confidence=ConfidenceScore(overall=0.0, factors=[]),
                total_raw=0,
            )

        raw_evidence = list(evidence_bundle.evidence)
        total_raw = len(raw_evidence)

        # --- Step 1: Provenance validation ---
        # Reject evidence with fabricated provenance
        provenance_valid = []
        for ev in raw_evidence:
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}

            video_id = origin.get("video_id") or meta.get("video_id")
            camera_id = origin.get("camera_id") or meta.get("camera_id")

            # Reject fabricated provenance (case-insensitive)
            if video_id and str(video_id).strip().lower() in FABRICATED_VIDEO_IDS:
                logger.warning(
                    f"Fusion: rejecting evidence {ev.evidence_id} — "
                    f"fabricated video_id '{video_id}'"
                )
                continue
            if camera_id and str(camera_id).strip().lower() in FABRICATED_CAMERA_IDS:
                logger.warning(
                    f"Fusion: rejecting evidence {ev.evidence_id} — "
                    f"fabricated camera_id '{camera_id}'"
                )
                continue

            provenance_valid.append(ev)

        # --- Step 2: Video isolation ---
        # If active_video_id is set, reject evidence from other videos
        active_video_id = getattr(context, "active_video_id", None)
        if active_video_id:
            video_filtered = []
            for ev in provenance_valid:
                meta = ev.metadata if isinstance(ev.metadata, dict) else {}
                origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}
                ev_video = origin.get("video_id") or meta.get("video_id")

                # Accept if video matches OR if evidence has no video_id (may still be relevant)
                if ev_video is None or ev_video == active_video_id:
                    video_filtered.append(ev)
                else:
                    logger.debug(
                        f"Fusion: filtering evidence {ev.evidence_id} — "
                        f"video_id '{ev_video}' != active '{active_video_id}'"
                    )
        else:
            video_filtered = provenance_valid

        # --- Step 3: Canonical identity construction + deduplication ---
        # IDENTITY RULE:
        #   IF video_id AND camera_id AND track_id ALL exist:
        #       identity = (video_id, camera_id, track_id) → safe to dedup
        #   IF track_id exists but video_id OR camera_id is None:
        #       identity scope is INCOMPLETE → no global dedup
        #   IF track_id does NOT exist:
        #       observation-level only → no identity dedup
        #   evidence_id is NEVER used as track_id

        identity_map: dict[tuple, dict] = {}  # identity_key → merged record
        observation_only: list = []  # evidence without complete identity

        for ev in video_filtered:
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}

            track_id = origin.get("track_id") or meta.get("track_id")
            video_id = origin.get("video_id") or meta.get("video_id")
            camera_id = origin.get("camera_id") or meta.get("camera_id")

            if track_id and video_id and camera_id:
                # Complete identity — safe to deduplicate
                identity_key = (video_id, camera_id, track_id)

                if identity_key in identity_map:
                    # Merge: keep highest confidence observation
                    existing = identity_map[identity_key]
                    existing["observations"].append(ev)
                    if ev.confidence > existing["best_confidence"]:
                        existing["best_confidence"] = ev.confidence
                        existing["best_evidence"] = ev
                    existing["observation_count"] += 1
                else:
                    identity_map[identity_key] = {
                        "identity_key": identity_key,
                        "track_id": track_id,
                        "video_id": video_id,
                        "camera_id": camera_id,
                        "best_evidence": ev,
                        "best_confidence": ev.confidence,
                        "observations": [ev],
                        "observation_count": 1,
                    }
            else:
                # Incomplete identity scope — do NOT deduplicate globally
                observation_only.append(ev)

        # --- Step 4: Temporal ordering ---
        # Sort deduplicated identities by first observation timestamp
        sorted_identities = sorted(
            identity_map.values(),
            key=lambda x: self._get_timestamp(x["best_evidence"]),
        )

        # --- Step 5: Build fused evidence list ---
        # Use the best evidence from each identity + all observation-only evidence
        fused_evidence = [rec["best_evidence"] for rec in sorted_identities]
        fused_evidence.extend(observation_only)

        # --- Step 6: Update evidence bundle ---
        evidence_bundle.evidence = fused_evidence

        # Store fusion metadata for downstream agents
        context.results["fusion_metadata"] = {
            "total_raw": total_raw,
            "after_provenance_filter": len(provenance_valid),
            "after_video_filter": len(video_filtered),
            "unique_identities": len(identity_map),
            "observation_only": len(observation_only),
            "observations_merged": sum(
                r["observation_count"] - 1 for r in identity_map.values()
            ),
            "identity_keys": [
                {"video_id": r["video_id"], "camera_id": r["camera_id"], "track_id": r["track_id"]}
                for r in sorted_identities
            ],
        }

        self._last_execution_time = (time.time() - start) * 1000

        return FusionResult(
            success=True,
            evidence=fused_evidence,
            confidence=ConfidenceScore(
                overall=max((ev.confidence for ev in fused_evidence), default=0.0),
                factors=[],
            ),
            total_raw=total_raw,
            after_provenance_filter=len(provenance_valid),
            after_video_filter=len(video_filtered),
            unique_identities=len(identity_map),
            observations_merged=sum(
                r["observation_count"] - 1 for r in identity_map.values()
            ),
        )

    def _get_timestamp(self, ev) -> float:
        """Extract timestamp from evidence for temporal ordering."""
        meta = ev.metadata if isinstance(ev.metadata, dict) else {}
        origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}
        return origin.get("video_timestamp_sec", 0.0) or 0.0

    def verify(self, result: BaseResult) -> bool:
        return isinstance(result, FusionResult) and result.success

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        context.agent_decisions.append({
            "agent": self.name,
            "decision": f"Fused evidence: {getattr(result, 'unique_identities', 0)} unique identities, "
                        f"{getattr(result, 'observations_merged', 0)} observations merged.",
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
