import pytest
from uuid import uuid4
from app.domain.memory.profile import MemoryProfile
from app.domain.memory.repository import InMemoryMemoryRepository
from app.domain.memory.events import MemoryEventBus
from app.domain.memory.manager import MemoryManager
from app.domain.memory.conversation import ConversationMemory, ConversationTurn
from app.domain.memory.entity import EntityMemory

def test_memory_lifecycle():
    # Setup Repositories
    repos = {
        "ConversationMemory": InMemoryMemoryRepository(),
        "EntityMemory": InMemoryMemoryRepository(),
        "FacilityMemory": InMemoryMemoryRepository(),
        "EpisodeMemory": InMemoryMemoryRepository(),
        "SummaryMemory": InMemoryMemoryRepository(),
        "InvestigationMemory": InMemoryMemoryRepository()
    }
    event_bus = MemoryEventBus()
    manager = MemoryManager(repositories=repos, event_bus=event_bus)
    
    # 1. Create Memories
    conv = ConversationMemory(session_id="123", turns=[ConversationTurn(speaker="USER", content="Hello")])
    ent = EntityMemory(entity_id=uuid4(), entity_type="Person", confidence=0.8)
    ent_high_conf = EntityMemory(entity_id=uuid4(), entity_type="Person", confidence=0.95)
    
    manager.add_memory(conv)
    manager.add_memory(ent)
    manager.add_memory(ent_high_conf)
    
    # 2. Retrieve & Rank & Inject
    # Planner with ITERATIVE profile
    planner_memories = manager.get_memories_for_agent("planner", MemoryProfile.ITERATIVE, {})
    assert len(planner_memories) == 1
    assert planner_memories[0].memory_type == "ConversationMemory"
    
    # Reasoning with ITERATIVE profile
    reasoning_memories = manager.get_memories_for_agent("reasoning", MemoryProfile.ITERATIVE, {})
    assert len(reasoning_memories) == 2
    # Ensure ranked by confidence
    assert reasoning_memories[0].confidence == 0.95
    assert reasoning_memories[1].confidence == 0.8
    assert all(m.memory_type == "EntityMemory" for m in reasoning_memories)

    # Reasoning with SIMPLE profile (should not fetch EntityMemory)
    simple_memories = manager.get_memories_for_agent("reasoning", MemoryProfile.SIMPLE, {})
    assert len(simple_memories) == 0
