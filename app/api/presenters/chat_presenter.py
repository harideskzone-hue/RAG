from app.api.schemas.response import ChatResponse, CitationModel, EvidenceModel
from app.schemas.context import VistaContext


class ChatPresenter:
    @staticmethod
    def present(canonical_response: dict, execution_id: str, processing_time_ms: int = 0) -> ChatResponse:
        # Use the final_answer from the canonical response, default to empty string if missing
        answer = canonical_response.get("final_answer")

        citations = []
        for c in canonical_response.get("citations", []):
            citations.append(CitationModel(
                source=c.get("source", ""),
                content=c.get("content", ""),
                confidence=c.get("confidence", 0.0)
            ))

        evidence = []
        for e in canonical_response.get("evidence", []):
            evidence.append(EvidenceModel(
                evidence_id=e["evidence_id"],
                source=e["source"],
                camera_id=e.get("camera_id"),
                timestamp=e.get("timestamp"),
                description=e.get("description"),
                confidence=e["confidence"]
            ))

        # Convert status to uppercase as expected by the API layer
        status = canonical_response.get("status", "SUCCESS")
        if status == "success":
            status = "SUCCESS"
        elif status == "error":
            status = "ERROR"
        # Keep other values as-is (like "blocked")

        return ChatResponse(
            status=status,
            answer=answer,
            confidence=canonical_response.get("overall_confidence", 0.0),
            citations=citations,
            evidence=evidence,
            processing_time_ms=processing_time_ms,
            trace_id=str(execution_id)
        )