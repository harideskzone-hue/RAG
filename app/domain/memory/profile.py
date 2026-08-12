from enum import Enum

class MemoryProfile(str, Enum):
    """
    Defines memory retrieval profiles tied to ExecutionModes.
    Determines exactly which types of memory the retriever should pull.
    """
    SIMPLE = "SIMPLE"                             # Conversation, Summary
    ITERATIVE = "ITERATIVE"                       # Conversation, Entity, Summary
    INVESTIGATION = "INVESTIGATION"               # Conversation, Entity, Episode, Facility, Summary
    DEEP_INVESTIGATION = "DEEP_INVESTIGATION"     # Everything including full Investigation logs
