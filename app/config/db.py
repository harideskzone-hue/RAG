import os
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    # PostgreSQL Configuration
    POSTGRES_URI: str = os.getenv("POSTGRES_URI", "postgresql+asyncpg://postgres:postgres@localhost:5433/vista")
    POSTGRES_POOL_SIZE: int = int(os.getenv("POSTGRES_POOL_SIZE", "10"))
    
    # MongoDB Configuration
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "vista_observations")
    
    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
    
    # Object Storage Configuration (Local initially)
    STORAGE_BASE_DIR: str = os.getenv("STORAGE_BASE_DIR", "./data/storage")

    @property
    def postgres_url(self) -> str:
        return self.POSTGRES_URI

    @property
    def mongo_uri(self) -> str:
        return self.MONGO_URI

    @property
    def mongo_db(self) -> str:
        return self.MONGO_DB_NAME

db_settings = DatabaseSettings()
