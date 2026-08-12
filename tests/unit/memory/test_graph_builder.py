from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.redis import AsyncRedisSaver

from app.graph.builder import GraphBuilder
from app.memory.manager import MemoryManager
from app.memory.policy import MemoryPolicy


@pytest.mark.asyncio
async def test_graph_builder_with_memory_and_redis():
    policy = MemoryPolicy()
    memory_manager = MemoryManager(policy)
    
    # Mock Redis checkpointer
    mock_redis = AsyncMock()
    # Checkpointer constructor might check for connection pool or something, we can mock it
    # Just passing a MagicMock to checkpointer in GraphBuilder is enough to test integration
    mock_checkpointer = MagicMock(spec=AsyncRedisSaver)
    
    builder = GraphBuilder(memory_manager=memory_manager, checkpointer=mock_checkpointer)
    builder.build_core_workflow()
    
    # Add dummy intent/planner/validator/supervisor nodes just for compilation
    builder.add_node("intent", lambda x: x)
    builder.add_node("planner", lambda x: x)
    builder.add_node("validator", lambda x: x)
    builder.add_node("supervisor", lambda x: x)
    
    # Now that nodes exist, add edges and compile
    builder.workflow.set_entry_point("intent")
    builder.add_edge("intent", "planner")
    builder.add_edge("planner", "validator")
    builder.add_edge("validator", "supervisor")
    
    # Memory manager edges
    builder.add_node("memory_manager", memory_manager.run)
    builder.add_edge("supervisor", "memory_manager")
    builder.add_edge("memory_manager", "__end__") # END is "__end__"
    
    app = builder.compile()
    assert app is not None
    assert app.checkpointer == mock_checkpointer
    
    # The nodes should be in the graph
    nodes = app.nodes
    assert "memory_manager" in nodes
