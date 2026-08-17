import os
import json
from abc import ABC, abstractmethod

from app.platform.config.config import config
from app.schemas.context import VistaContext


class CheckpointStore(ABC):
    @abstractmethod
    async def save(self, context: VistaContext) -> None:
        pass


class JSONCheckpointStore(CheckpointStore):
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dataset")
        self.data_dir = data_dir
        self.checkpoints_path = os.path.join(data_dir, "checkpoints.json")
        os.makedirs(self.data_dir, exist_ok=True)
        self._checkpoints = {}
        self._load()

    def _load(self):
        if os.path.exists(self.checkpoints_path):
            try:
                with open(self.checkpoints_path, 'r') as f:
                    self._checkpoints = json.load(f)
            except Exception as e:
                import logging
                logging.warning(f"Failed to load checkpoints: {e}")

    def _save(self):
        tmp_path = self.checkpoints_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(self._checkpoints, f)
        os.replace(tmp_path, self.checkpoints_path)

    async def save(self, context: VistaContext) -> None:
        # Save by conversation_id
        self._checkpoints[context.conversation_id] = context.model_dump_json()
        self._save()


class RedisCheckpointStore(CheckpointStore):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        import redis.asyncio as redis
        self.pool = redis.ConnectionPool.from_url(redis_url)
        self.client = redis.Redis(connection_pool=self.pool)

    async def save(self, context: VistaContext) -> None:
        state_data = context.model_dump_json()
        await self.client.set(f"vista_context:{context.conversation_id}", state_data)

    async def close(self) -> None:
        await self.client.close()
        await self.pool.disconnect()


def get_checkpoint_store() -> CheckpointStore:
    if config.mode == "native":
        return JSONCheckpointStore()
    return RedisCheckpointStore(config.redis_url)
