from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.security.jwt import JWTService
from app.security.permissions import has_permission
from app.security.roles import Role

security = HTTPBearer(auto_error=False)

def get_jwt_service() -> JWTService:
    return JWTService()

def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_service: JWTService = Depends(get_jwt_service)
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header"
        )
    try:
        return jwt_service.verify_token(credentials.credentials)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

def require_permission(permission: str):
    def permission_checker(current_user: dict = Depends(get_current_user)):
        role_str = current_user.get("role")
        try:
            role = Role(role_str)
            if not has_permission(role, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role_str}' does not have permission '{permission}'"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unknown role '{role_str}'"
            )
        return current_user
    return permission_checker
