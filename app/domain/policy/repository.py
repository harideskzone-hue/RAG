from typing import Protocol
from app.domain.policy.rules import PolicyRule

class PolicyRepository(Protocol):
    """Decouples policy loading from the Engine."""
    def load_rules(self, policy_id: str) -> list[PolicyRule]:
        ...

class InMemoryPolicyRepository(PolicyRepository):
    """Basic in-memory storage for MVP, allowing quick rule iteration."""
    def __init__(self):
        self._policies: dict[str, list[PolicyRule]] = {}
        
    def save_policy(self, policy_id: str, rules: list[PolicyRule]):
        self._policies[policy_id] = rules
        
    def load_rules(self, policy_id: str) -> list[PolicyRule]:
        return self._policies.get(policy_id, [])
