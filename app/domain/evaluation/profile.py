from enum import Enum

class EvaluationProfile(str, Enum):
    """Allows selective benchmarking of subsystems."""
    FAST = "FAST"           # Planner, Supervisor, Policy
    FULL = "FULL"           # Everything
    REASONING = "REASONING" # Reasoning only
    GRAPH = "GRAPH"         # Knowledge Graph only
