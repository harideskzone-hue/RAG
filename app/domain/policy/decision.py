from enum import Enum

class PolicyDecision(str, Enum):
    APPROVE = "APPROVE"
    MODIFY = "MODIFY"
    DEFER = "DEFER"
    REJECT = "REJECT"
