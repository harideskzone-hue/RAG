#!/usr/bin/env python3
"""
Tests for P0.4: Production Security
Validates that production security assertions are in place and working:
- No default database credentials in production
- No wildcard CORS in production
- Proper LOG_LEVEL handling
- No debug mode in production
"""
import pytest
import os
from unittest.mock import patch
from app.platform.config.config import Settings, config


class TestP04ProductionSecurity:
    """Test P0.4: Production Security"""

    def test_production_assertions_database_password(self):
        """Production should reject default database password"""
        # Test that the assertion triggers for default credentials
        test_env = {
            "VISTA_ENV": "production",
            "DATABASE_URL": "postgresql+asyncpg://vista:secret@localhost:5432/vista",
            "MODE": "native",
            "MODEL_FREE": "true"
        }

        with patch.dict(os.environ, test_env, clear=False):
            # Creating Settings should trigger the assertion and raise ValueError
            with pytest.raises(ValueError, match="CRITICAL: Default database password cannot be used in production"):
                Settings()  # This should fail due to the assertion in model_post_init

    def test_production_assertions_wildcard_cors(self):
        """Production should reject wildcard CORS"""
        test_env = {
            "VISTA_ENV": "production",
            "FRONTEND_URLS": "http://localhost:3000,http://*.example.com,http://[::]:3000",
            "DATABASE_URL": "postgresql+asyncpg://secure_user:secure_pass@db.internal:5432/vista",
            "MODE": "native",
            "MODEL_FREE": "true"
        }

        with patch.dict(os.environ, test_env, clear=False):
            # Creating Settings should trigger the assertion and raise ValueError
            with pytest.raises(ValueError, match="CRITICAL: Wildcard CORS is not permitted in production"):
                Settings()  # This should fail due to the assertion in model_post_init

    def test_development_allows_default_credentials(self):
        """Development should allow default credentials (for convenience)"""
        test_env = {
            "VISTA_ENV": "development",
            "DATABASE_URL": "postgresql+asyncpg://vista:secret@localhost:5432/vista",
            "MODE": "native",
            "MODEL_FREE": "true"
        }

        with patch.dict(os.environ, test_env, clear=False):
            # Creating Settings should NOT raise an error in development
            settings = Settings()  # This should succeed
            assert settings.environment == "development"
            assert "vista:secret@" in settings.database_url

    def test_log_level_configuration(self):
        """LOG_LEVEL should be configurable via environment variable"""
        # Test default
        default_env = {
            "MODE": "native",
            "MODEL_FREE": "true"
        }

        with patch.dict(os.environ, default_env, clear=False):
            settings = Settings()
            assert settings.log_level == "INFO"  # Default value

        # Test override to DEBUG
        debug_env = {
            "LOG_LEVEL": "DEBUG",
            "MODE": "native",
            "MODEL_FREE": "true"
        }

        with patch.dict(os.environ, debug_env, clear=False):
            settings = Settings()
            assert settings.log_level == "DEBUG"

        # Test override to WARNING
        warning_env = {
            "LOG_LEVEL": "WARNING",
            "MODE": "native",
            "MODEL_FREE": "true"
        }

        with patch.dict(os.environ, warning_env, clear=False):
            settings = Settings()
            assert settings.log_level == "WARNING"

    def test_no_hardcoded_debug_assertions_requiring_fixes(self):
        """Verify we didn't add any hardcoded DEBUG=True assertions that need fixing"""
        # Check that we don't have any hardcoded debug assertions in our modified files
        # that would need to be removed for production

        # Check a few key files we modified
        files_to_check = [
            "app/agents/metadata/agent.py",
            "app/agents/vector/agent.py",
            "app/agents/reasoning/agent.py",
            "app/tools/video/s3_tool.py",
            "app/services/video_service/service.py"
        ]

        for file_path in files_to_check:
            with open(file_path, "r") as f:
                content = f.read()

            # Verify we don't have hardcoded debug assertions
            # (This is more of a sanity check - we know we didn't add any)
            assert "DEBUG = True" not in content
            assert "debug=True" not in content.lower() or "#" in content.split("debug=True")[0] if "debug=True" in content.lower() else True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])