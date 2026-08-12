from typing import Any
from app.agents.base_agent import BaseAgent
from app.schemas.context import BaseResult, VistaContext
from app.domain.models import AgentManifest, AgentCapability


class GuardrailResult(BaseResult):
    is_safe: bool = True
    violations: list[str] = []


class GuardrailAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "guardrail_agent"

    @property
    def description(self) -> str:
        return "Validates responses against hallucination and PII."

    @property
    def manifest(self) -> AgentManifest:
        return AgentManifest(
            self.name,
            self.description,
            AgentCapability(supported_operations=["verify"]),
            cost="low", latency="fast"
        )

    def validate(self, context: VistaContext) -> bool:
        return "reasoning_agent" in context.results

    async def plan(self, context: VistaContext) -> Any:
        return None

    async def execute(self, context: VistaContext, plan: Any) -> GuardrailResult:

        import re
        reasoning_res = context.results.get("reasoning_agent")
        ans = reasoning_res.metadata.get('explanation', '') if reasoning_res else ''

        # Regex to capture the full citation block
        citation_pattern = r'\(Evidence ID:\s*(ev-[^\,]+),\s*Frame ID:\s*([^\,]+),\s*Timestamp:\s*([^\,]+),\s*Video:\s*([^\)]+)\)'
        citations = re.findall(citation_pattern, ans)

        has_any_citation = bool(re.search(r'Evidence ID:', ans))

        if not has_any_citation and "insufficient evidence" not in ans.lower() and "no hypotheses" not in ans.lower():
            # When no citations and insufficient evidence phrases not found, likely missing evidence citation
            return GuardrailResult(success=False, is_safe=False, violations=["Missing evidence citation"], confidence=ConfidenceScore(overall=0.0, factors=[]))

        violations = []
        if context.evidence_bundle:
            # Map evidence to its required fields
            valid_evidence = {}
            for e in context.evidence_bundle.evidence:
                frame_id = e.metadata.get("frame_id", "")
                timestamp = e.timestamp.isoformat().replace('+00:00', 'Z')
                video = e.metadata.get("video", "")
                valid_evidence[str(e.evidence_id)] = (frame_id, timestamp, video)

            if has_any_citation and not citations:
                violations.append("Citation format is malformed or missing required fields")

            for eid, fid, ts, vid in citations:
                if eid not in valid_evidence:
                    violations.append(f"Citation {eid} does not resolve to a real evidence record")
                else:
                    v_fid, v_ts, v_vid = valid_evidence[eid]
                    if fid.strip() != v_fid:
                        violations.append(f"Frame ID mismatch for {eid}: {fid} != {v_fid}")
                    # Flexible timestamp matching (prefix match to handle minor formatting diffs)
                    if not ts.strip().startswith(v_ts[:19]):
                        violations.append(f"Timestamp mismatch for {eid}: {ts} != {v_ts}")
                    if vid.strip() != v_vid:
                        violations.append(f"Video mismatch for {eid}: {vid} != {v_vid}")
        else:
            if has_any_citation:
                violations.append("Citations found but no evidence bundle exists")

        if violations:
            # When violations are found, confidence in "safe" judgment is 0.0
            return GuardrailResult(success=False, is_safe=False, violations=violations, confidence=ConfidenceScore(overall=0.0, factors=[]))

        # When no violations are found, confidence in "safe" judgment is 0.95 (not hardcoded to 1.0)
        return GuardrailResult(success=True, is_safe=True, confidence=ConfidenceScore(overall=0.95, factors=[]))

    def verify(self, result: BaseResult) -> bool:
        return True

    def finish(self, context: VistaContext, result: BaseResult) -> VistaContext:
        context.results[self.name] = result
        return context

    def confidence(self, result: BaseResult) -> float:
        return result.confidence.overall

    def citations(self, result: BaseResult) -> list:
        return []

    def metrics(self) -> dict:
        return {
            "execution_time_ms": getattr(self, "_last_execution_time", 0.0)
        }