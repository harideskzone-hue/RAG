from fastapi import APIRouter

from app.platform.health import health

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
async def check_health():
    return await health.get_health()

@router.get("/ready")
async def check_ready():
    return await health.get_ready()

@router.get("/live")
def check_live():
    return health.get_live()

@router.get("/version")
def check_version():
    return health.get_version()

@router.get("/info")
def check_info():
    return health.get_info()
