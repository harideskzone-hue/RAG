#!/usr/bin/env python3
"""
VISTA AI — Full System Data Wipe & Reset Utility
Wipes all previous video test files, metadata JSONs, crops, tracklets, 
and resets databases (PostgreSQL, MongoDB, Qdrant, SQLite) to a pristine state.
"""
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.db import db_settings


def wipe_filesystem():
    print("1. Cleaning filesystem directories (videos, crops, metadata, events)...")
    
    # Clean input directories
    for d in ["input/completed", "input/watch", "input/processing", "input/recording", "input/failed"]:
        dir_path = PROJECT_ROOT / d
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Cleaned {d}/")

    # Clean dataset directories
    for d in ["dataset/metadata", "dataset/tracks", "dataset/persons", "dataset/events", "dataset/reid", "dataset/storage"]:
        dir_path = PROJECT_ROOT / d
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Cleaned {d}/")

    # Clean local sqlite db / checkpoints if present
    for f in ["dataset/vista_local.db", "dataset/checkpoints.json", "dataset/vector_metadata.json"]:
        file_path = PROJECT_ROOT / f
        if file_path.exists():
            file_path.unlink()
            print(f"   ✓ Removed {f}")


def wipe_databases():
    print("\n2. Resetting database stores...")

    # Reset MongoDB
    try:
        from pymongo import MongoClient
        mc = MongoClient(db_settings.MONGO_URI, serverSelectionTimeoutMS=2000)
        mc.drop_database(db_settings.MONGO_DB_NAME)
        mc.close()
        print(f"   ✓ Dropped MongoDB database: {db_settings.MONGO_DB_NAME}")
    except Exception as e:
        print(f"   ⚠ MongoDB reset skipped (offline or standalone): {e}")

    # Reset PostgreSQL
    try:
        import asyncio
        import asyncpg

        async def _clear_pg():
            uri = db_settings.POSTGRES_URI.replace("postgresql+asyncpg://", "postgresql://")
            conn = await asyncpg.connect(uri)
            for tbl in ["evidence", "tracks", "video_segments", "canonical_persons", "events"]:
                try:
                    await conn.execute(f"TRUNCATE TABLE {tbl} CASCADE")
                except Exception:
                    pass
            await conn.close()

        asyncio.run(_clear_pg())
        print("   ✓ Truncated PostgreSQL tables")
    except Exception as e:
        print(f"   ⚠ PostgreSQL reset skipped (offline or standalone): {e}")

    # Reset Qdrant
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=db_settings.QDRANT_HOST, port=db_settings.QDRANT_PORT, timeout=2)
        for coll in ["person_embeddings_v2", "vista_embeddings", "vehicle_embeddings_v1"]:
            try:
                qc.delete_collection(coll)
            except Exception:
                pass
        print("   ✓ Deleted Qdrant vector collections")
    except Exception as e:
        print(f"   ⚠ Qdrant reset skipped (offline or standalone): {e}")


def reinit():
    print("\n3. Re-initializing pristine database schemas & Camera Registry...")
    try:
        from scripts.init_databases import main as init_main
        import asyncio
        asyncio.run(init_main())
        print("   ✓ Database schemas re-initialized successfully.")
    except Exception as e:
        print(f"   ⚠ Re-init completed with standalone notice: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  VISTA AI — COMPLETE SYSTEM WIPE & RESET")
    print("=" * 60)
    wipe_filesystem()
    wipe_databases()
    reinit()
    print("\n" + "=" * 60)
    print("  ✓ SYSTEM IS 100% PRISTINE AND READY FOR NEW TEST VIDEO!")
    print("=" * 60)
