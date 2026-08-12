
from app.domain.models import Alert
from app.schemas.context import VistaContext
from app.tools.base_tool import BaseTool


class AlertRepository:
    """
    Hides SQL queries for alerts. Returns typed domain models.
    """
    def __init__(self, postgres_tool: BaseTool):
        self.db = postgres_tool

    async def get_alerts_by_camera(self, camera_id: str, context: VistaContext) -> list[Alert]:
        if context.user and context.user.allowed_cameras and camera_id not in context.user.allowed_cameras:
            return []
            
        # Use parameterized query to prevent SQL injection
        query = "SELECT * FROM alerts WHERE camera_id = $1"
        result = await self.db.execute(context, query=query, params=[camera_id])
        
        if result.success and result.rows:
            return [Alert(**row) for row in result.rows]
        return []

    async def get_recent_alerts(self, limit: int, context: VistaContext) -> list[Alert]:
        if context.user and context.user.allowed_cameras:
            placeholders = ", ".join(f"${i+2}" for i in range(len(context.user.allowed_cameras)))
            query = f"SELECT * FROM alerts WHERE camera_id IN ({placeholders}) ORDER BY timestamp DESC LIMIT $1"
            params = [limit] + context.user.allowed_cameras
        else:
            query = "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT $1"
            params = [limit]
            
        result = await self.db.execute(context, query=query, params=params)
        
        if result.success and result.rows:
            return [Alert(**row) for row in result.rows]
        return []

