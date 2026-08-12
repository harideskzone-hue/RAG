import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.platform.errors.security import SecurityError
from app.security.roles import Role

# Secret key used to sign the JWT tokens.
_env = os.getenv("VISTA_ENV", "development")
_secret = os.getenv("JWT_SECRET_KEY")
if _env == "production" and not _secret:
    raise ValueError("CRITICAL: JWT_SECRET_KEY must be set in production.")
SECRET_KEY = _secret or "super_secret_local_dev_key"
ALGORITHM = "HS256"

class JWTService:
    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(hours=24)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            role: str = payload.get("role")
            if user_id is None or role is None:
                raise SecurityError("Invalid token payload: missing sub or role")
            # Validate role exists
            try:
                Role(role)
            except ValueError:
                raise SecurityError(f"Invalid role in token: {role}")
            return payload
        except JWTError as e:
            raise SecurityError(f"Could not validate credentials: {e!s}")
