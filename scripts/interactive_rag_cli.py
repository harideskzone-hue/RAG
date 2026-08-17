import os
import sys
import logging
import asyncio
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure environment before loading registry
os.environ["CV_MODEL_DIR"] = "/Users/hariharans/Documents/longgraph/models"
os.environ["CV_DETECTOR_MODEL"] = "yolo26n.pt"
os.environ["CV_DEVICE"] = "cpu"
os.environ["CV_SAMPLE_FPS"] = "5"

from app.cv.pipeline.video_pipeline import VideoPipeline
from app.agents.evidence_fusion.agent import EvidenceFusionAgent
from app.agents.verification.agent import VerificationAgent
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.schemas.context import VistaContext, UserContext
from app.domain.evidence import VideoEvidence, EvidenceBundle
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING)

def wrap_contract_to_evidence(contract) -> VideoEvidence:
    """Wraps CV EvidenceContract into RAG BaseEvidence for fusion pipeline."""
    return VideoEvidence(
        evidence_id=contract.evidence_id,
        source=contract.source,
        confidence=contract.confidence,
        timestamp=datetime.now(timezone.utc),
        metadata={
            "track_id": contract.subject.track_id,
            "video_id": contract.provenance.video_id,
            "camera_id": contract.provenance.camera_id,
            "bbox": contract.observation.get("bbox")
        }
    )

async def run_interactive_cli(video_path: str):
    print("========================================")
    print("      VISTA End-to-End Interactive CLI")
    print("========================================")
    print(f"Loading and processing video: {video_path}")
    print("This will take a moment as CV extracts all evidence...")
    
    pipeline = VideoPipeline()
    contracts = pipeline.process_video(video_path=video_path, video_id="VID_TEST", camera_id="CAM_TEST")
    
    unique_tracks = set([c.subject.track_id for c in contracts if c.subject.track_id])
    print(f"✅ CV Pipeline Complete. Extracted {len(contracts)} observations across {len(unique_tracks)} tracker identities.")
    
    fusion_agent = EvidenceFusionAgent()
    verification_agent = VerificationAgent()
    coordinator = ResponseCoordinator()
    
    print("\n" + "="*40)
    print("Interactive Chatbot Ready.")
    print("Ask questions about the video (e.g., 'How many people are in the video?')")
    print("Type 'exit' to quit.")
    print("="*40)

    while True:
        try:
            query = input("\n[You]: ")
            if query.strip().lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue

            # Mock intent parsing for now since live API intent classification is bypassed in local manual test
            class MockIntentResult:
                class QueryIntent:
                    operation = "count" if "how many" in query.lower() else "search"
                    target_type = "person"
                    semantic_constraints = []
                    attributes = []
                query_intent = QueryIntent()

            user_ctx = UserContext(user_id="cli_user", role="admin")
            context = VistaContext(user=user_ctx)
            context.active_video_id = "VID_TEST"
            context.current_query = query
            
            # Wrap contracts and push as EvidenceBundle
            base_evidences = [wrap_contract_to_evidence(c) for c in contracts]
            context.evidence_bundle = EvidenceBundle(evidence=base_evidences)
            context.results["intent_agent"] = MockIntentResult()

            # Execute RAG Loop
            await fusion_agent.execute(context, None)
            await verification_agent.execute(context, None)
            response = coordinator.generate_response(context)

            print("\n[VISTA AI]:")
            print(response.get("final_answer", "No answer could be generated."))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.10 interactive_rag_cli.py <path_to_video>")
        sys.exit(1)
    asyncio.run(run_interactive_cli(sys.argv[1]))
