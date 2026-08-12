
from app.domain.models import VehicleMatch
from app.schemas.context import VistaContext
from app.services.repositories.base_repository import BaseRepository


class VehicleRepository(BaseRepository):
    """
    Hides Milvus internal collection details for vehicle search.
    Returns typed domain models.
    """
    
    async def search_vehicle(self, embedding: list[float], top_k: int, context: VistaContext) -> list[VehicleMatch]:
        collection_name = "vehicle_embeddings_v1"
        
        result = await self.tool.execute(context, collection=collection_name, embedding=embedding, top_k=top_k)
        
        matches = []
        if result.success and result.matches:
            for row in result.matches:
                if hasattr(row, 'model_dump'):
                    matches.append(VehicleMatch(**row.model_dump()))
                else:
                    matches.append(VehicleMatch(**row))
                
        return matches
