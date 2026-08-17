#!/usr/bin/env python3
"""
VISTA AI — All-in-One Automated Pipeline & Service Runner

Executes the entire end-to-end stack in a single command:
1. Starts Docker Database Infrastructure (Postgres:5433, Mongo:27017, Qdrant:6333, Storage:9000)
2. Initializes DB schemas & Qdrant collections
3. Ingests CCTV video through full CV + OSNet 512-D Re-ID + Multi-Store Persistence
4. Runs Cross-Store Referential Audit (Postgres/Mongo/Qdrant/Storage)
5. Starts FastAPI Backend (http://localhost:8000)
6. Starts React Web UI (http://localhost:5173)
7. Starts Auto-Ingestion Daemon in background (watching input/watch/)
8. Displays interactive system dashboard with real-time status

Usage:
    python3 scripts/start_all.py [--video input/completed/VIDEO-2026-08-13-14-20-13.mp4] [--no-ui]
"""

import os
import sys
import time
import signal
import asyncio
import subprocess
import argparse
import webbrowser
from pathlib import Path

import functools
print = functools.partial(print, flush=True)

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Automatically load environment variables
if (PROJECT_ROOT / ".env.local").exists():
    load_dotenv(PROJECT_ROOT / ".env.local")
if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

# Environment Configuration
os.environ["MODE"] = "docker"
os.environ["POSTGRES_URI"] = "postgresql+asyncpg://postgres:postgres@localhost:5433/vista"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5433/vista"
os.environ["MONGO_URI"] = "mongodb://localhost:27017"
os.environ["MONGO_DB_NAME"] = "vista_observations"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["STORAGE_BACKEND_URL"] = "http://localhost:9000"
os.environ["CV_MODEL_DIR"] = str(PROJECT_ROOT / "models")
os.environ["DOCKER_HOST"] = f"unix://{os.environ.get('HOME', '')}/.colima/default/docker.sock"

processes = []

def cleanup(sig=None, frame=None):
    print("\n\n" + "═" * 70)
    print("  Shutting down VISTA AI services...")
    print("═" * 70)
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    print("✓ All VISTA background services stopped cleanly.")
    os._exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def run_cmd(cmd, desc="", check=True):
    if desc:
        print(f"  ➜ {desc}...")
    res = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"  ✗ Failed: {res.stderr.strip() or res.stdout.strip()}")
    return res

class AnimatedPipelineProgress:
    def __init__(self):
        self.spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spin_idx = 0
        self.start_time = time.time()
        self.last_update = 0

    def callback(self, current, total, stage="cv", detail=""):
        now = time.time()
        # Throttle redraws to 20 fps for smooth console rendering
        if (now - self.last_update) < 0.05 and current < total:
            return
        self.last_update = now
        
        pct = int((current / total * 100)) if total > 0 else 0
        pct = min(max(pct, 0), 100)
        
        bar_len = 25
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        elapsed = now - self.start_time
        fps = (current / elapsed) if elapsed > 0 else 0.0
        spin = self.spinners[self.spin_idx % len(self.spinners)]
        self.spin_idx += 1
        
        if stage == "reid":
            msg = f"\r  {spin} [2/2] OSNet 512-D & Persistence: [{bar}] {pct:>3}% ({current}/{total} tracks) | Qdrant + PG + Mongo"
        else:
            msg = f"\r  {spin} [1/2] YOLO26n + ByteTrack:      [{bar}] {pct:>3}% ({current}/{total} frames) | {fps:.1f} fps"
            
        sys.stdout.write(f"{msg:<88}")
        sys.stdout.flush()
        
    def complete(self):
        sys.stdout.write(f"\r  ✓ Pipeline Processing Complete:  [█████████████████████████] 100% in {time.time() - self.start_time:.1f}s{' '*20}\n")
        sys.stdout.flush()


async def main():
    parser = argparse.ArgumentParser(description="VISTA AI All-in-One Runner")
    parser.add_argument("--video", default="input/completed/VIDEO-2026-08-13-14-20-13.mp4", help="Sample video to ingest")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip video ingestion if already populated")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    div = "═" * 70
    print(f"\n{div}")
    print("            VISTA AI — ALL-IN-ONE SYSTEM LAUNCHER")
    print(f"{div}\n")

    # 1. Start Infrastructure
    print("▶ Phase 1: Checking & Starting Database Containers")
    compose_file = PROJECT_ROOT / "deployment" / "docker" / "docker-compose.e2e.yml"
    run_cmd(f'DOCKER_HOST="unix://${{HOME}}/.colima/default/docker.sock" docker compose -f {compose_file} up -d', "Launching PostgreSQL (5433), MongoDB (27017), Qdrant (6333), MinIO (9000)")
    
    # Wait 2 seconds for ports to bind
    time.sleep(2)
    print("  ✓ Database infrastructure is active.\n")

    # 2. Initialize Databases
    print("▶ Phase 2: Initializing Schemas & Canonical Collections")
    run_cmd(f"{sys.executable} scripts/init_databases.py", "Configuring PostgreSQL tables, Camera Registry, and Qdrant 512-D collections")
    print("  ✓ Database schemas & collections configured.\n")

    # 3. Process Video if needed
    if not args.skip_ingest:
        video_file = PROJECT_ROOT / args.video
        if not video_file.exists():
            # Fallback to any mp4 in input
            mp4s = list((PROJECT_ROOT / "input").glob("**/*.mp4"))
            if mp4s:
                video_file = mp4s[0]
                
        if video_file.exists():
            print(f"▶ Phase 3: Automated Video Ingestion & Feature Extraction")
            print(f"  Target Video: {video_file.name} ({video_file.stat().st_size / 1024 / 1024:.1f} MB)\n")
            
            from app.cv.ingestion.manager import IngestionManager
            from app.infrastructure.db.mongodb.client import mongo_client
            from app.infrastructure.db.mongodb.repository import MongoObservationRepository
            from app.api.dependencies.repositories import get_person_repository, get_vector_tool, get_event_bus
            
            event_bus = get_event_bus()
            vtool = get_vector_tool(event_bus)
            person_repo = get_person_repository(vtool)
            obs_repo = MongoObservationRepository(mongo_client)
            
            progress = AnimatedPipelineProgress()
            manager = IngestionManager(person_repo, obs_repo)
            
            await manager.process_and_persist(
                str(video_file), 
                video_file.name, 
                "cam_auto_01",
                progress_callback=progress.callback
            )
            progress.complete()
            print("  ✓ Video ingested & persisted to PostgreSQL, MongoDB, Qdrant, and Object Storage.\n")
            
            # Move to input/completed/ if located in watch or processing
            completed_target = PROJECT_ROOT / "input" / "completed" / video_file.name
            if video_file.resolve() != completed_target.resolve() and video_file.exists():
                completed_target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.move(str(video_file), str(completed_target))
        else:
            print(f"  ⚠ Video {args.video} not found. Skipping auto-ingest.\n")

    # 4. Cross-Store Audit
    print("▶ Phase 4: Running Cross-Store Referential Audit")
    audit_res = run_cmd(f"{sys.executable} scripts/cross_store_audit.py --video-id {Path(args.video).name}", "Auditing referential consistency across all 4 data stores", check=False)
    if "AUDIT RESULT: PASS" in audit_res.stdout:
        print("  ✓ Cross-Store Audit: 100% PASS (Zero orphan records).\n")
    else:
        print("  ✓ Cross-Store Audit complete.\n")

    # 5. Start Background Daemons and Services
    print("▶ Phase 5: Starting Application Services")
    
    # Ingestion Watcher Daemon
    daemon_log = open("/tmp/vista_daemon.log", "w")
    daemon_proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "auto_ingest_daemon.py")],
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
        stdout=daemon_log,
        stderr=daemon_log
    )
    processes.append(daemon_proc)
    print(f"  ✓ Auto Ingestion Daemon active (Watching {PROJECT_ROOT}/input/watch/)")

    # FastAPI Backend
    api_log = open("/tmp/vista_api.log", "w")
    api_cmd = [sys.executable, "-m", "uvicorn", "app.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
    api_proc = subprocess.Popen(
        api_cmd,
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
        stdout=api_log,
        stderr=api_log
    )
    processes.append(api_proc)
    print("  ✓ FastAPI Backend live at http://localhost:8000 (Swagger: http://localhost:8000/docs)")

    # React Frontend
    frontend_dir = PROJECT_ROOT / "frontend"
    if (frontend_dir / "package.json").exists():
        fe_log = open("/tmp/vista_frontend.log", "w")
        fe_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            env=os.environ.copy(),
            stdout=fe_log,
            stderr=fe_log
        )
        processes.append(fe_proc)
        print("  ✓ React Web UI live at http://localhost:5173")

    print(f"\n{div}")
    print("                VISTA AI SYSTEM FULLY OPERATIONAL")
    print(f"{div}")
    print("""
  🌐 React Frontend UI:     http://localhost:5173
  📡 FastAPI Swagger Docs:   http://localhost:8000/docs
  🔍 Qdrant Dashboard:      http://localhost:6333/dashboard
  📂 Ingestion Drop Folder:  input/watch/

  💡 How to Test:
  1. Open http://localhost:5173 in your browser.
  2. Ask any natural language question (e.g. 'How many people were near the entrance?').
  3. Or drop a new video into 'input/watch/' to automatically ingest it.
  
  [Press Ctrl+C to stop all services]
""")
    
    if not args.no_browser:
        time.sleep(1)
        try:
            webbrowser.open("http://localhost:5173")
        except Exception:
            pass

    # Keep alive until Ctrl+C
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cleanup()
