from typing import Protocol, TypeVar, Generic, Any
from uuid import UUID

T = TypeVar('T')

class MemoryRepository(Protocol, Generic[T]):
    """Protocol for abstracting Memory persistence (Redis, Postgres, InMemory)."""
    def save(self, memory_id: str | UUID, memory: T) -> None: ...
    def load(self, memory_id: str | UUID) -> T | None: ...
    def delete(self, memory_id: str | UUID) -> None: ...
    def search(self, **kwargs) -> list[T]: ...

class InMemoryMemoryRepository(MemoryRepository[Any]):
    """In-Memory implementation of the MemoryRepository for Phase 3."""
    def __init__(self):
        self._storage: dict[str, Any] = {}
        
    def save(self, memory_id: str | UUID, memory: Any) -> None:
        self._storage[str(memory_id)] = memory
        
    def load(self, memory_id: str | UUID) -> Any | None:
        return self._storage.get(str(memory_id))
        
    def delete(self, memory_id: str | UUID) -> None:
        self._storage.pop(str(memory_id), None)
        
    def search(self, **kwargs) -> list[Any]:
        # Very simple mocked search for now.
        return list(self._storage.values())
