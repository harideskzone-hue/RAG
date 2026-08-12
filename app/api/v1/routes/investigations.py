from fastapi import APIRouter, Depends

from app.api.dependencies.security import require_permission

router = APIRouter(prefix="/investigations", tags=["investigations"])

@router.get("", dependencies=[Depends(require_permission("read:report"))])
def list_investigations():
    return {"investigations": []}
