import os
import pytest

def test_production_config_validation():
    original_env = os.environ.get("VISTA_ENV")
    original_db = os.environ.get("DATABASE_URL")
    original_frontend = os.environ.get("FRONTEND_URLS")
    
    try:
        os.environ["VISTA_ENV"] = "production"
        
        # Test Default Password Rejection
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://vista:secret@localhost:5432/vista"
        
        import sys
        if "app.platform.config.config" in sys.modules:
            del sys.modules["app.platform.config.config"]
            
        with pytest.raises(ValueError, match="CRITICAL: Default database password cannot be used in production."):
            import app.platform.config.config
            
        # Test Wildcard CORS rejection
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://vista:secure_prod_pwd@localhost:5432/vista"
        os.environ["FRONTEND_URLS"] = "*"
        
        if "app.platform.config.config" in sys.modules:
            del sys.modules["app.platform.config.config"]
            
        with pytest.raises(ValueError, match="CRITICAL: Wildcard CORS is not permitted in production."):
            import app.platform.config.config
            
        # Valid config should pass
        os.environ["FRONTEND_URLS"] = "https://vista.example.com"
        
        if "app.platform.config.config" in sys.modules:
            del sys.modules["app.platform.config.config"]
            
        import app.platform.config.config
        assert app.platform.config.config.config.environment == "production"
        
    finally:
        # Restore environment variables
        if original_env is not None: os.environ["VISTA_ENV"] = original_env
        else: os.environ.pop("VISTA_ENV", None)
            
        if original_db is not None: os.environ["DATABASE_URL"] = original_db
        else: os.environ.pop("DATABASE_URL", None)
            
        if original_frontend is not None: os.environ["FRONTEND_URLS"] = original_frontend
        else: os.environ.pop("FRONTEND_URLS", None)
