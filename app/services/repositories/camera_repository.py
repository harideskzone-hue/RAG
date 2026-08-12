
from app.domain.models import Camera
from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool


class CameraRepository:
    """
    Hides SQL queries. Returns typed domain models.
    """
    def __init__(self, postgres_tool: BaseTool):
        self.db = postgres_tool

    async def get_camera(self, camera_id: str, context: VistaContext) -> Camera | None:
        if context.user and context.user.allowed_cameras and camera_id not in context.user.allowed_cameras:
            return None
            
        # Use parameterized query to prevent SQL injection
        query = "SELECT * FROM cameras WHERE id = $1"
        result = await self.db.execute(context, query=query, params=[camera_id])
        
        if result.success and result.rows:
            row = result.rows[0]
            return Camera(**row)
        return None

    async def get_all_cameras(self, context: VistaContext) -> list[Camera]:
        if context.user and context.user.allowed_cameras:
            placeholders = ", ".join(f"${i+1}" for i in range(len(context.user.allowed_cameras)))
            query = f"SELECT * FROM cameras WHERE id IN ({placeholders})"
            params = context.user.allowed_cameras
        else:
            query = "SELECT * FROM cameras"
            params = []
            
        result = await self.db.execute(context, query=query, params=params)
        
        if result.success and result.rows:
            return [Camera(**row) for row in result.rows]
        return []

