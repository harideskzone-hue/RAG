import os
import aioboto3
from abc import ABC, abstractmethod
from typing import Any
import asyncio

from app.platform.config.config import config


class BlobStore(ABC):
    @abstractmethod
    async def get_uri(self, bucket: str, key: str) -> tuple[str, float]:
        pass
        
    @abstractmethod
    async def upload(self, bucket: str, key: str, file_path: str) -> None:
        pass

    @abstractmethod
    async def health(self) -> bool:
        pass


class LocalFileStore(BlobStore):
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dataset", "storage")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    async def get_uri(self, bucket: str, key: str) -> tuple[str, float]:
        bucket_dir = os.path.join(self.storage_dir, bucket)
        file_path = os.path.join(bucket_dir, key)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local blob not found: {file_path}")
            
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        uri = f"file://{file_path}"
        return uri, size_mb

    async def upload(self, bucket: str, key: str, file_path: str) -> None:
        bucket_dir = os.path.join(self.storage_dir, bucket)
        os.makedirs(bucket_dir, exist_ok=True)
        dest_path = os.path.join(bucket_dir, key)
        
        # Use asyncio to thread copy
        import shutil
        await asyncio.to_thread(shutil.copy2, file_path, dest_path)

    async def health(self) -> bool:
        return True


class S3Store(BlobStore):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    async def get_uri(self, bucket: str, key: str) -> tuple[str, float]:
        session = aioboto3.Session()
        async with session.client('s3', endpoint_url=self.endpoint_url, aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin') as s3:
            head = await s3.head_object(Bucket=bucket, Key=key)
            size_mb = head['ContentLength'] / (1024 * 1024)
            uri = await s3.generate_presigned_url('get_object',
                                                  Params={'Bucket': bucket, 'Key': key},
                                                  ExpiresIn=3600)
            return uri, size_mb
            
    async def upload(self, bucket: str, key: str, file_path: str) -> None:
        session = aioboto3.Session()
        async with session.client('s3', endpoint_url=self.endpoint_url, aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin') as s3:
            try:
                await s3.head_bucket(Bucket=bucket)
            except:
                await s3.create_bucket(Bucket=bucket)
            with open(file_path, 'rb') as f:
                await s3.put_object(Bucket=bucket, Key=key, Body=f)

    async def health(self) -> bool:
        try:
            session = aioboto3.Session()
            async with session.client('s3', endpoint_url=self.endpoint_url, aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin') as s3:
                await s3.list_buckets()
                return True
        except Exception:
            return False


def get_blob_store() -> BlobStore:
    if config.mode == "native":
        return LocalFileStore()
    return S3Store(config.storage_backend_url)
