import asyncio
import json
import logging
from app.api.dependencies.repositories import get_postgres_tool, get_milvus_tool
from app.api.dependencies.services import (
    get_event_bus, get_s3_tool,
    get_metadata_service, get_vector_service, get_video_service,
    get_event_service, get_report_service
)
from app.api.dependencies.supervisor import _initialize_registries
from app.agents.vector.agent import VectorAgent
from app.schemas.context import VistaContext, UserContext, QueryIntent
from app.agents.intent.classifier import IntentResult

logging.basicConfig(level=logging.INFO)

async def main():
    eb = get_event_bus()
    pt = get_postgres_tool()
    mt = get_milvus_tool()
    st = get_s3_tool()
    meta_svc = get_metadata_service(pt)
    vec_svc = get_vector_service(mt)
    vid_svc = get_video_service(st)
    evt_svc = get_event_service(pt)
    rpt_svc = get_report_service(pt)

    _initialize_registries(eb, pt, mt, st, meta_svc, vec_svc, vid_svc, evt_svc, rpt_svc)

    user = UserContext(user_id="admin_user", role="admin", allowed_cameras=["cam_01", "cam_02"])
    context = VistaContext(user=user, current_query="how many men in the cctv", active_video_id="VIDEO-2026-08-13-14-20-13.mp4")

    qi = QueryIntent(
        domain="investigation",
        operation="count",
        target_type="person",
        semantic_constraints=["gender=male"],
        attributes=["gender=male"],
        search_operations=["vector_person"]
    )
    context.results["intent_agent"] = IntentResult(
        success=True,
        intent="count",
        domain="investigation",
        operation="count",
        entities={"description": "how many men in the cctv"},
        query_intent=qi
    )

    agent = VectorAgent(vec_svc)
    res = await agent.execute(context, None)

    print("VectorResult Status:", res.status)
    print("VectorResult Error Metadata:", res.metadata.get("error"))
    print("Person Matches Count:", len(res.person_matches))
    for p in res.person_matches:
        print(f"  Match ID={p.id[:8]}... | Cam={p.camera_id} | Track={p.origin.get('track_id')} | Gender={p.attributes.get('gender')}")

if __name__ == "__main__":
    asyncio.run(main())
