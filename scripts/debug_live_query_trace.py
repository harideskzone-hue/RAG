import asyncio
import json
import logging
from app.api.dependencies.repositories import get_postgres_tool, get_milvus_tool
from app.api.dependencies.services import (
    get_event_bus, get_s3_tool,
    get_metadata_service, get_vector_service, get_video_service,
    get_event_service, get_report_service
)
from app.api.dependencies.supervisor import _initialize_registries, Supervisor
from app.schemas.context import VistaContext, UserContext

logging.basicConfig(level=logging.INFO)

async def test_all_queries():
    eb = get_event_bus()
    pt = get_postgres_tool()
    mt = get_milvus_tool()
    st = get_s3_tool()
    meta_svc = get_metadata_service(pt)
    vec_svc = get_vector_service()
    vid_svc = get_video_service(st)
    evt_svc = get_event_service(pt)
    rpt_svc = get_report_service(pt)

    _initialize_registries(eb, pt, mt, st, meta_svc, vec_svc, vid_svc, evt_svc, rpt_svc)

    queries = [
        ("how many men in the cctv", "Count Men"),
        ("how many women in the cctv", "Count Women"),
        ("how many kids?", "Count Kids"),
        ("any fire accident?", "Event Search Fire")
    ]

    print("=" * 80)
    print("🔬 RUNNING LIVE PIPELINE E2E TRACE FOR ALL 4 QUERIES")
    print("=" * 80)

    for q, label in queries:
        print(f"\n📌 QUERY: [{label}] \"{q}\"")
        user = UserContext(user_id="admin_user", role="admin", allowed_cameras=["cam_01", "cam_02"])
        context = VistaContext(user=user, current_query=q, active_video_id="VIDEO-2026-08-13-14-20-13.mp4")

        supervisor = Supervisor()
        supervisor.event_bus = eb
        res_dict = await supervisor.run(context)

        contract = context.results.get("verified_contract", {})
        print(f"  • Status: {contract.get('status')}")
        print(f"  • Operation: {contract.get('operation')}")
        print(f"  • Verified Count: {contract.get('verified_count')}")
        print(f"  • Verified Tracks: {contract.get('verified_tracks')}")
        print(f"  • Final Answer: \"{res_dict.get('final_answer')}\"")
        print(f"  • Formatted Evidence Cards Count: {len(res_dict.get('evidence', []))}")

if __name__ == "__main__":
    asyncio.run(test_all_queries())
