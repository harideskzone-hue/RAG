import motor.motor_asyncio
import pymongo
from app.config.db import db_settings

class MongoDBClient:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(db_settings.MONGO_URI)
        self.db = self.client[db_settings.MONGO_DB_NAME]
        self.observations_col = self.db["observations"]

    async def setup_indexes(self):
        """
        Create necessary indexes for temporal and identity retrieval.
        - evidence_id
        - video_id
        - camera_id + timestamp_sec
        - track_id + timestamp_sec
        """
        # Ensure evidence_id uniqueness per observation (if 1:1, or index if 1:N)
        await self.observations_col.create_index([("evidence_id", pymongo.ASCENDING)])
        
        # Fast filtering by video
        await self.observations_col.create_index([("video_id", pymongo.ASCENDING)])
        
        # Temporal filtering by camera
        await self.observations_col.create_index([
            ("camera_id", pymongo.ASCENDING),
            ("timestamp_sec", pymongo.ASCENDING)
        ])
        
        # Temporal filtering by track identity
        await self.observations_col.create_index([
            ("track_id", pymongo.ASCENDING),
            ("timestamp_sec", pymongo.ASCENDING)
        ])

mongo_client = MongoDBClient()
