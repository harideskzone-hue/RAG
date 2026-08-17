#!/usr/bin/env python3
"""
VISTA AI Database Initialization Utility
Creates tables and collections across PostgreSQL and Qdrant.
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine
from app.infrastructure.db.postgres.models import Base, CameraModel
from app.config.db import db_settings
from app.schemas.evidence_contract import CameraRegistry
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("InitDB")


async def init_postgres():
    logger.info("Initializing PostgreSQL schema...")
    engine = create_async_engine(db_settings.POSTGRES_URI, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    logger.info("PostgreSQL schema created successfully.")
    
    # Insert default cameras
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.dialects.postgresql import insert
    
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        for cam_id in ["cam_01", "cam_auto_01", "entrance_cam", "exit_cam"]:
            CameraRegistry.register(cam_id)
            stmt = insert(CameraModel).values(
                camera_id=cam_id,
                location="Main Entrance",
                resolution="1920x1080"
            ).on_conflict_do_nothing(index_elements=["camera_id"])
            await session.execute(stmt)
        await session.commit()
    logger.info("Default cameras registered in PostgreSQL and CameraRegistry.")
    await engine.dispose()


def init_qdrant():
    logger.info("Initializing Qdrant collections...")
    client = QdrantClient(host=db_settings.QDRANT_HOST, port=db_settings.QDRANT_PORT, timeout=5)
    
    collections_to_create = ["person_embeddings_v2", "vista_embeddings", "vehicle_embeddings_v1"]
    existing = [c.name for c in client.get_collections().collections]
    
    for coll in collections_to_create:
        if coll not in existing:
            client.create_collection(
                collection_name=coll,
                vectors_config=qmodels.VectorParams(size=512, distance=qmodels.Distance.COSINE)
            )
            client.create_payload_index(coll, "entity_id", qmodels.PayloadSchemaType.KEYWORD)
            client.create_payload_index(coll, "camera_id", qmodels.PayloadSchemaType.KEYWORD)
            client.create_payload_index(coll, "video_id", qmodels.PayloadSchemaType.KEYWORD)
            logger.info(f"Created Qdrant collection: {coll} (dim=512, cosine)")
        else:
            logger.info(f"Qdrant collection {coll} already exists.")


async def main():
    try:
        await init_postgres()
    except Exception as e:
        logger.error(f"Failed to init PostgreSQL: {e}")
        
    try:
        init_qdrant()
    except Exception as e:
        logger.error(f"Failed to init Qdrant: {e}")


if __name__ == "__main__":
    asyncio.run(main())
