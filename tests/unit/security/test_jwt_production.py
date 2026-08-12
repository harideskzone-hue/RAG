import os
import pytest

def test_production_jwt_requires_secret():
    # Store original environment variables
    original_env = os.environ.get("VISTA_ENV")
    original_secret = os.environ.get("JWT_SECRET_KEY")
    
    try:
        # Simulate production environment without secret
        os.environ["VISTA_ENV"] = "production"
        if "JWT_SECRET_KEY" in os.environ:
            del os.environ["JWT_SECRET_KEY"]
            
        import importlib
        import sys
        if "app.security.jwt" in sys.modules:
            del sys.modules["app.security.jwt"]
            
        with pytest.raises(ValueError, match="CRITICAL: JWT_SECRET_KEY must be set in production."):
            import app.security.jwt as jwt_module
            
        # Add secret, it should pass
        os.environ["JWT_SECRET_KEY"] = "prod_secret_123"
        import app.security.jwt as jwt_module
        assert jwt_module.SECRET_KEY == "prod_secret_123"
        
    finally:
        # Restore environment variables
        if original_env is not None:
            os.environ["VISTA_ENV"] = original_env
        else:
            del os.environ["VISTA_ENV"]
            
        if original_secret is not None:
            os.environ["JWT_SECRET_KEY"] = original_secret
        elif "JWT_SECRET_KEY" in os.environ:
            del os.environ["JWT_SECRET_KEY"]
