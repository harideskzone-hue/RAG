import os
import sys
import logging
import asyncio
from uuid import uuid4
from unittest.mock import MagicMock

# Environment for CV
os.environ["CV_MODEL_DIR"] = "/Users/hariharans/Documents/longgraph/models"
os.environ["CV_DETECTOR_MODEL"] = "yolo26n.pt"
os.environ["CV_DEVICE"] = "cpu"
os.environ["CV_SAMPLE_FPS"] = "5"

from app.cv.pipeline.video_pipeline import VideoPipeline
from app.agents.evidence_fusion.agent import EvidenceFusionAgent
from app.agents.verification.agent import VerificationAgent
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.schemas.context import VistaContext, UserContext

logging.basicConfig(level=logging.WARNING)

async def run_e2e_mutation(video_path: str):
    print("========================================")
    print("      VISTA E2E RAG Mutation Test")
    print("========================================")
    
    pipeline = VideoPipeline()
    contracts = pipeline.process_video(video_path=video_path, video_id="VID_TEST", camera_id="CAM_TEST")
    
    unique_tracks = set([c.subject.track_id for c in contracts if c.subject.track_id])
    original_track_count = len(unique_tracks)
    print(f"[CV Pipeline] Extracted {len(contracts)} observations across {original_track_count} tracker identities.")
    
    # -----------------------------------------------------------------
    # RAG RUN 1 (Original)
    # -----------------------------------------------------------------
    user_ctx = UserContext(user_id="cli_user", role="admin")
    context = VistaContext(user=user_ctx)
    context.active_video_id = "VID_TEST"
    
    # Simulate intent extraction for a count query
    class MockIntentResult:
        class QueryIntent:
            operation = "count"
            target_type = "person"
            semantic_constraints = []
            attributes = []
        query_intent = QueryIntent()
    context.results["intent_agent"] = MockIntentResult()
    context.current_query = "How many people are in the video?"
    
    fusion_agent = EvidenceFusionAgent()
    verification_agent = VerificationAgent()
    coordinator = ResponseCoordinator()
    
    # Push contracts as initial findings
    context.findings = contracts
    
    await fusion_agent.execute(context, None)
    await verification_agent.execute(context, None)
    resp1 = await coordinator.execute(context, None)
    
    print("\n[RAG Pipeline - Run 1]")
    print(f"Verified Count: {context.results.get('verified_contract').verified_count}")
    print(f"Final Answer:   {resp1.get('final_answer')}")
    
    # -----------------------------------------------------------------
    # RAG RUN 2 (Mutation: Drop half the tracks)
    # -----------------------------------------------------------------
    drop_tracks = list(unique_tracks)[:original_track_count // 2]
    mutated_contracts = [c for c in contracts if c.subject.track_id not in drop_tracks]
    
    mutated_track_count = len(set([c.subject.track_id for c in mutated_contracts if c.subject.track_id]))
    print(f"\n[Mutation] Dropped {len(drop_tracks)} tracks. Remaining: {mutated_track_count} tracker identities.")
    
    context2 = VistaContext(user=user_ctx)
    context2.active_video_id = "VID_TEST"
    context2.results["intent_agent"] = MockIntentResult()
    context2.current_query = "How many people are in the video?"
    context2.findings = mutated_contracts
    
    await fusion_agent.execute(context2, None)
    await verification_agent.execute(context2, None)
    resp2 = await coordinator.execute(context2, None)
    
    print("\n[RAG Pipeline - Run 2]")
    print(f"Verified Count: {context2.results.get('verified_contract').verified_count}")
    print(f"Final Answer:   {resp2.get('final_answer')}")
    
    print("========================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.10 run_rag_e2e_mutation.py <path_to_video>")
        sys.exit(1)
    asyncio.run(run_e2e_mutation(sys.argv[1]))
