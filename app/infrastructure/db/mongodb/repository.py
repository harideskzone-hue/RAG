from typing import Any, List
from uuid import UUID

from app.domain.repositories.base import ObservationRepository
from app.infrastructure.db.mongodb.client import MongoDBClient

class MongoObservationRepository(ObservationRepository):
    def __init__(self, client: MongoDBClient):
        self.client = client

    async def insert_observation(self, observation: dict[str, Any]) -> None:
        """
        Upserts the observation to guarantee idempotency.
        """
        await self.client.observations_col.update_one(
            {"evidence_id": observation["evidence_id"]},
            {"$set": observation},
            upsert=True
        )

    async def get_observations_for_evidence(self, evidence_id: str | UUID) -> List[dict[str, Any]]:
        cursor = self.client.observations_col.find({"evidence_id": str(evidence_id)})
        return await cursor.to_list(length=None)
