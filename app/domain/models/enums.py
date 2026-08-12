from enum import Enum

class SchemaVersion(str, Enum):
    V1_0 = "1.0"
    V1_1 = "1.1"

class AgentType(str, Enum):
    METADATA = "metadata"
    VECTOR = "vector"
    VIDEO = "video"
    EVENT = "event"
    REPORT = "report"
    REASONING = "reasoning"

class AgentStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    ERROR = "error"

class ExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionMode(str, Enum):
    SIMPLE = "simple"
    ITERATIVE = "iterative"
    INVESTIGATION = "investigation"

class EntityType(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    CAMERA = "camera"
    ZONE = "zone"
    EVENT = "event"
    OBJECT = "object"
    SCENE = "scene"
    INCIDENT = "incident"
    ALERT = "alert"
    REPORT = "report"
    UNKNOWN = "unknown"

class RelationshipType(str, Enum):
    SEEN_AT = "seen_at"
    ENTERED = "entered"
    EXITED = "exited"
    OWNS = "owns"
    FOLLOWED_BY = "followed_by"
    CO_OCCURRED = "co_occurred"
    CONFIRMS_PRESENCE_ON = "confirms_presence_on"
    PART_OF = "part_of"
    IDENTITY = "identity"
    UNKNOWN = "unknown"

class GraphHint(str, Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    IDENTITY = "identity"
    TRACKING = "tracking"
    PROXIMITY = "proximity"
    MOVEMENT = "movement"
    OWNERSHIP = "ownership"

class EvidenceType(str, Enum):
    METADATA = "metadata"
    VECTOR = "vector"
    VIDEO = "video"
    EVENT = "event"
    REPORT = "report"
    LLM = "llm"

class ReasoningStage(str, Enum):
    CORRELATION = "correlation"
    CONTRADICTION = "contradiction"
    GAP_ANALYSIS = "gap_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    HYPOTHESIS_RANKING = "hypothesis_ranking"
    EXPLANATION = "explanation"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"
