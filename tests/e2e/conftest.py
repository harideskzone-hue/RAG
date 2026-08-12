import pytest
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app.app import app
from app.security.jwt import JWTService
from app.security.roles import Role


@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture(scope="session")
def valid_token():
    jwt_service = JWTService()
    return jwt_service.create_access_token({"sub": "e2e_user", "role": Role.OPERATOR})

@pytest.fixture(scope="session")
def admin_token():
    jwt_service = JWTService()
    return jwt_service.create_access_token({"sub": "admin_user", "role": Role.ADMIN})

@pytest.fixture(scope="session")
def viewer_token():
    jwt_service = JWTService()
    return jwt_service.create_access_token({"sub": "viewer_user", "role": Role.VIEWER})
