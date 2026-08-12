from typing import Protocol, List, Dict, Any

class VLMResponse:
    def __init__(self, content: str):
        self.content = content

class VLMClient(Protocol):
    async def ainvoke(self, messages: List[Dict[str, Any]]) -> VLMResponse:
        ...
