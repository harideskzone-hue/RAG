#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import uuid
import time

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.platform.config.config import config
config.mode = "native"

from app.schemas.context import VistaContext, UserContext
from app.graph.supervisor.supervisor import Supervisor
from app.graph.supervisor.event_bus import EventBus
from app.tools.metadata.postgres_tool import PostgresTool
from app.tools.vector.vector_tool import VectorTool
from app.tools.video.s3_tool import S3Tool
from app.services.metadata_service import MetadataService
from app.services.vector_service import VectorService
from app.services.video_service.service import VideoService
from app.services.event_service.service import EventService
from app.services.report_service.service import ReportService
from app.services.repositories.camera_repository import CameraRepository
from app.services.repositories.alert_repository import AlertRepository
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository
from app.services.video_service.vlm_adapter import GeminiAdapter
from app.api.dependencies.supervisor import _initialize_registries
from app.api.presenters.chat_presenter import ChatPresenter
from scripts.ingest_video import ingest_video


async def run_live_e2e_validation():
    div = "=" * 70
    print(div)
    print("🚀 VISTA Live End-to-End Pipeline & Provenance Validation")
    print(div)
    
    # 1. Ingest real MP4 footage to build real dataset with origin + attributes
    video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "VIDEO-2026-08-13-14-20-13.mp4")
    video_id = "VIDEO-2026-08-13-14-20-13.mp4"
    ingest_video(video_path, video_id=video_id, camera_id="cam_01")
    
    # 2. Initialize supervisor & service registries
    event_bus = EventBus()
    postgres_tool = PostgresTool(event_bus)
    vector_tool = VectorTool(event_bus)
    s3_tool = S3Tool(event_bus)
    
    metadata_service = MetadataService(CameraRepository(postgres_tool), AlertRepository(postgres_tool), event_bus)
    vector_service = VectorService(PersonRepository(vector_tool), VehicleRepository(vector_tool), event_bus)
    video_service = VideoService(s3_tool, GeminiAdapter(), event_bus)
    event_service = EventService(event_bus)
    report_service = ReportService(event_bus)
    
    _initialize_registries(
        event_bus, postgres_tool, vector_tool, s3_tool,
        metadata_service, vector_service, video_service,
        event_service, report_service
    )
    
    supervisor = Supervisor()
    supervisor.event_bus = event_bus
    
    queries = [
        "How many people are in the video?",
        "How many men are there?",
        "How many women are there?",
        "Is there any suspicious person in the CCTV?",
        "Tell me about the person wearing the royal blue fleece jacket.",
        "Where did this person go?"
    ]
    
    conv_id = f"conv_e2e_{int(time.time())}"
    
    for i, query in enumerate(queries, 1):
        print(f"\n{div}")
        print(f"📌 Query #{i}: \"{query}\"")
        print(f"{div}")
        
        supervisor = Supervisor()
        supervisor.event_bus = event_bus
        
        context = VistaContext(
            user=UserContext(user_id="e2e_tester", role="admin"),
            conversation_id=conv_id,
            execution_id=str(uuid.uuid4()),
            current_query=query,
            active_video_id=video_id
        )
        
        start_time = time.time()
        result = await supervisor.run(context)
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Present through API ChatPresenter
        presenter_output = ChatPresenter.present(result, context.execution_id, elapsed_ms)
        
        print("\n🔍 --- 1. Query Intent & Constraints ---")
        intent_agent = context.results.get("intent_agent")
        if intent_agent:
            qi = getattr(intent_agent, "query_intent", None)
            if qi:
                print(f"   Domain: {getattr(qi, 'domain', '')}")
                print(f"   Operation: {getattr(qi, 'operation', '')}")
                print(f"   Target: {getattr(qi, 'target_type', '')}")
                print(f"   Semantic Constraints: {getattr(qi, 'semantic_constraints', [])}")
                print(f"   Attributes: {getattr(qi, 'attributes', [])}")

        print("\n🛡️ --- 2. Provenance Gate & Evidence Contract ---")
        contract = context.results.get("verified_contract", {})
        print(f"   Contract Status: {contract.get('status', 'N/A')}")
        print(f"   Verified Count: {contract.get('verified_count', 0)}")
        print(f"   Verified Tracks: {contract.get('verified_tracks', [])}")
        
        print(f"\n📦 --- 3. Retrieved Evidence Cards ({len(presenter_output.evidence)} items) ---")
        for ev in presenter_output.evidence:
            print(f"   • Card: UUID={ev.evidence_id[:12]}... | Camera={ev.camera_id} | Source={ev.source}")
            print(f"     Description: {ev.description}")
            
        print("\n⚡ --- 4. Dynamic Execution Telemetry ---")
        for step in presenter_output.execution.steps:
            print(f"   • [{step.name}]: status={step.status}, latency={step.latency_ms}ms")
            
        print("\n💬 --- 5. Natural Language Response ---")
        print(f"{presenter_output.answer}")
        
    print(f"\n{div}")
    print("✅ Live End-to-End Validation Complete!")
    print(f"{div}\n")

if __name__ == "__main__":
    asyncio.run(run_live_e2e_validation())
