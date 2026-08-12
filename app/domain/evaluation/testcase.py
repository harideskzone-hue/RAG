from pydantic import BaseModel
from typing import Any

class TestCase(BaseModel):
    __test__ = False
    test_id: str
    description: str
    query: str
    expected_agents: list[str] = []
    expected_entities: list[str] = []
    expected_relationships: list[str] = []
    metadata: dict[str, Any] = {}
