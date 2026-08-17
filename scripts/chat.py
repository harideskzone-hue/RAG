#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import uuid
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.evidence import PersonEvidence, VehicleEvidence, MetadataEvidence, VideoEvidence
from app.tools.ingestors.cv_ingestor import load_mock_cv_inputs
from app.tools.vector.store import get_vector_store
from app.tools.vector.encoder import get_vector_encoder
from app.tools.metadata.store import get_metadata_store
from app.schemas.context import VistaContext
from app.domain.models.enums import ExecutionMode
from app.graph.supervisor.supervisor import Supervisor
from app.domain.llm.ollama_reasoning_client import OllamaReasoningClient
from app.platform.config.config import config

# Import agents to ensure they are registered
from app.agents.intent.classifier import HybridIntentClassifier
from app.agents.planner.planner import ExecutionPlanner
from app.agents.metadata.agent import MetadataAgent
from app.agents.vector.agent import VectorAgent
from app.agents.video.agent import VideoAgent
from app.agents.event.agent import EventAgent
from app.agents.evidence.agent import EvidenceAgent
from app.agents.reasoning.agent import ReasoningAgent
from app.agents.report.agent import ReportAgent

logger = logging.getLogger(__name__)

async def seed_stores(fixtures_dir: str):
    print("\n--- Seeding Local Stores from Mock CV Inputs ---")
    # Load bundle
    bundle = load_mock_cv_inputs(fixtures_dir)
    print(f"Loaded {len(bundle.evidence)} evidence items from {fixtures_dir}")

    # Seed Vector Store
    vector_store = get_vector_store()
    encoder = get_vector_encoder()

    # Reset vector store to ensure clean slate
    vector_store.vectors = __import__('numpy').array([])
    vector_store.metadata = []

    ids, embs, cams, times, descs, bboxes = [], [], [], [], [], []
    for ev in bundle.evidence:
        if isinstance(ev, (PersonEvidence, VehicleEvidence)):
            ids.append(str(ev.evidence_id))
            text_desc = ev.metadata.get("description", "")
            
            # Simple attribute serialization for embedding
            if isinstance(ev, PersonEvidence):
                attrs = ev.metadata.get("attributes", {})
                if attrs:
                    text_desc += " " + json.dumps(attrs)
            elif isinstance(ev, VehicleEvidence):
                attrs = ev.metadata.get("attributes", {})
                if attrs:
                    text_desc += " " + json.dumps(attrs)

            embs.append(encoder.encode(text_desc))
            cams.append(ev.metadata.get("camera_id", ""))
            times.append(ev.timestamp.isoformat())
            descs.append(text_desc)
            bboxes.append(ev.metadata.get("bbox"))

    if ids:
        await vector_store.insert("test_collection", [ids, embs, cams, times, descs, bboxes])
        print(f"Seeded {len(ids)} items into NativeVectorStore")

    # Seed Metadata Store (SQLite)
    meta_store = get_metadata_store()
    await meta_store.execute("DELETE FROM alerts")
    await meta_store.execute("DELETE FROM cameras")
    
    # Insert cameras from statistics or general knowledge
    cam_id = bundle.statistics.get("camera_id", "cam_01")
    await meta_store.execute(
        "INSERT INTO cameras (id, location, status, firmware_version) VALUES (?, ?, ?, ?)",
        cam_id, "Mock Location", "online", "v1.0"
    )

    alert_count = 0
    for ev in bundle.evidence:
        if isinstance(ev, MetadataEvidence):
            await meta_store.execute(
                "INSERT INTO alerts (id, camera_id, type, severity, timestamp) VALUES (?, ?, ?, ?, ?)",
                str(ev.evidence_id), 
                ev.metadata.get("camera_id", cam_id),
                ev.metadata.get("event_type", "event"),
                ev.metadata.get("severity", "info"),
                ev.timestamp.isoformat()
            )
            alert_count += 1
    
    print(f"Seeded 1 camera and {alert_count} alerts into SQLiteMetadataStore")
    print("------------------------------------------------\n")


async def chat_loop(mode: str):
    llm_client = OllamaReasoningClient()
    
    # Initialize registries
    from app.api.dependencies.supervisor import _initialize_registries
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
    
    supervisor = Supervisor(llm_client=llm_client)
    supervisor.event_bus = event_bus
    
    print("VISTA AI Agentic RAG Pipeline Ready.")
    print(f"Mode: {mode}")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("Query> ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if query.strip().lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue

        start_time = time.time()
        from app.schemas.context import UserContext
        context = VistaContext(
            user=UserContext(user_id="cli_user", role="admin"),
            conversation_id=str(uuid.uuid4()),
            execution_id=str(uuid.uuid4()),
            current_query=query,
            execution_mode=ExecutionMode.INVESTIGATION if mode.lower() == "investigation" else ExecutionMode.SIMPLE
        )

        print("\n--- Executing Pipeline ---")
        try:
            response = await supervisor.run(context)
            
            # Print trace overview
            print("\n--- Execution Trace ---")
            intent_agent = context.results.get("intent_agent")
            if intent_agent:
                print(f"Intent: {getattr(intent_agent, 'query_intent', 'UNKNOWN')}")
            
            print(f"Execution Plan: {[group for group in getattr(context.execution_plan, 'execution_groups', [])]}")
            
            # Print ledger errors
            for record in context.execution_ledger:
                if record.status == "failed" and record.error:
                    print(f"Agent {record.agent_name} failed: {record.error}")

            evidence_count = len(response.get("evidence", []))
            print(f"Evidence Retrieved: {evidence_count} items")
            
            print(f"\nFinal Answer:\n{response.get('final_answer', response.get('content', 'No content'))}")
            
            print(f"\nConfidence: {response.get('overall_confidence', response.get('metadata', {}).get('confidence_score', 0.0))}")
            print(f"Latency: {(time.time() - start_time):.2f}s")
            
        except Exception as e:
            import traceback
            print(f"\nPipeline Error: {e}")
            traceback.print_exc()
        
        print("\n" + "="*50 + "\n")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Interactive Agentic RAG tests against Mock CV data.")
    parser.add_argument("--mode", type=str, choices=["simple", "investigation"], default="simple", help="Execution mode")
    parser.add_argument("--fixtures", type=str, default="dataset/mock_cv_inputs", help="Path to mock CV JSONs")
    args = parser.parse_args()

    # Ensure config points to native mode for testing
    config.mode = "native"

    try:
        await seed_stores(args.fixtures)
        await chat_loop(args.mode)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Disable overly verbose httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
