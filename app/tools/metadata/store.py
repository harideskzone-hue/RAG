import os
import sqlite3
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Any

from app.platform.config.config import config
import asyncpg


class MetadataStore(ABC):
    @abstractmethod
    async def execute(self, query: str, *args) -> list[dict]:
        pass

    @abstractmethod
    async def health(self) -> bool:
        pass


class SQLiteMetadataStore(MetadataStore):
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dataset", "vista_local.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_metadata (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT,
                    timestamp TEXT,
                    description TEXT,
                    entities_json TEXT,
                    video_uri TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    location TEXT,
                    status TEXT,
                    firmware_version TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT,
                    type TEXT,
                    severity TEXT,
                    timestamp TEXT
                )
            """)

    def _execute(self, query: str, args: tuple) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Basic translation from postgres to sqlite if needed, but we keep it simple
            query = query.replace("$1", "?").replace("$2", "?").replace("$3", "?").replace("$4", "?").replace("$5", "?").replace("$6", "?")
            cur = conn.execute(query, args)
            if query.strip().upper().startswith("SELECT"):
                rows = cur.fetchall()
                return [dict(row) for row in rows]
            return []

    async def execute(self, query: str, *args) -> list[dict]:
        return await asyncio.to_thread(self._execute, query, args)

    async def health(self) -> bool:
        return True


class PostgresMetadataStore(MetadataStore):
    def __init__(self, db_url: str):
        self.db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    async def execute(self, query: str, *args) -> list[dict]:
        conn = await asyncpg.connect(self.db_url)
        try:
            if query.strip().upper().startswith("SELECT"):
                records = await conn.fetch(query, *args)
                return [dict(record) for record in records]
            else:
                await conn.execute(query, *args)
                return []
        finally:
            await conn.close()

    async def health(self) -> bool:
        try:
            conn = await asyncpg.connect(self.db_url)
            await conn.close()
            return True
        except Exception:
            return False


def get_metadata_store() -> MetadataStore:
    if config.mode == "native":
        return SQLiteMetadataStore()
    return PostgresMetadataStore(config.database_url)
