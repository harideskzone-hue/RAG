
from app.domain.models import PersonMatch
from app.schemas.context import VistaContext
from app.services.repositories.base_repository import BaseRepository


class PersonRepository(BaseRepository):
    """
    Hides Milvus internal collection details for person search.
    Returns typed domain models.
    """
    
    async def search_person(self, embedding: list[float], top_k: int, context: VistaContext) -> list[PersonMatch]:
        # Collection name isolated here
        collection_name = "person_embeddings_v2"
        
        result = await self.tool.execute(context, collection=collection_name, embedding=embedding, top_k=top_k)
        
        matches = []
        if result.success and result.matches:
            for row in result.matches:
                if hasattr(row, 'model_dump'):
                    matches.append(PersonMatch(**row.model_dump()))
                else:
                    matches.append(PersonMatch(**row))
                
        return matches
