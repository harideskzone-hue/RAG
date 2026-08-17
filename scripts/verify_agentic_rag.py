#!/usr/bin/env python3
"""
Comprehensive Agentic RAG End-to-End Verification Test
Tests Intent Classification -> Multi-Store Semantic Retrieval -> VerifiedResultContract -> LLM Reasoning -> Grounding Validator -> Response Presenter
"""
import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.context import VistaContext, UserContext
from app.api.dependencies.supervisor import get_supervisor
from app.api.presenters.chat_presenter import ChatPresenter


async def run_query(query: str, video_id: str = "chain_robbery_cctv.mp4"):
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    
    user = UserContext(user_id="investigator_1", role="admin", allowed_cameras=["cam_auto_01"])
    context = VistaContext(
        user=user,
        conversation_id="test_forensic_rag",
        current_query=query,
        active_video_id=video_id
    )
    
    supervisor = get_supervisor()
    res = await supervisor.run(context)
    pres = ChatPresenter.present(res, "exec_test_001")
    
    print(f"Status:            {pres.status}")
    print(f"Detection Status:  {pres.detection_status}")
    print(f"Person Count:      {pres.person_count}")
    print(f"Zone:              {pres.zone}")
    print(f"Evaluation Window: {pres.evaluation_window}")
    print(f"Confidence:        {pres.confidence:.2f}")
    print(f"Grounding Status:  {pres.grounding_status}")
    print(f"\nAnswer:\n{pres.answer}")
    if pres.thinking_process:
        print(f"\nThinking Process:\n{pres.thinking_process}")
    print(f"\nEvidence Verified: {len(pres.evidence)} items")
    for i, ev in enumerate(pres.evidence[:3], 1):
        print(f"  [{i}] ID: {ev.evidence_id} | Person: {ev.person_id} | Track: {ev.track_id} | Crop: {ev.crop_url}")
        print(f"      Desc: {ev.description}")
    print(f"\nExecution Steps: {[s.name for s in pres.execution.steps]}")
    print("=" * 70 + "\n")


async def main():
    queries = [
        "How many people are visible in this video?",
        "What activities and events occurred in this CCTV footage?",
        "Show me the suspect or individual near the entrance."
    ]
    for q in queries:
        await run_query(q)


if __name__ == "__main__":
    asyncio.run(main())
