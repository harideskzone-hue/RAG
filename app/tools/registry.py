from collections.abc import Callable


class ToolRegistry:
    """
    Central registry for all external tools.
    The Supervisor requests tools through this registry.
    """
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, tool_func: Callable, description: str) -> None:
        if name in self._tools:
            raise ValueError(f"Tool with name {name} is already registered.")
        self._tools[name] = tool_func
        self._descriptions[name] = description

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def get_all_tools(self) -> dict[str, Callable]:
        return self._tools.copy()
        
    def get_tool_descriptions(self) -> dict[str, str]:
        return self._descriptions.copy()

# Singleton registry
tool_registry = ToolRegistry()
