from typing import List, Tuple
from enum import Enum

class ResolutionStatus(Enum):
    MATCHED = "MATCHED"
    NEW = "NEW"
    UNRESOLVED = "UNRESOLVED"

class IdentityResolver:
    """
    Evaluates Qdrant Top-K similarity search results to decide if a new embedding
    belongs to an existing canonical person, is a completely new person, or is ambiguous.
    """
    def __init__(self, 
                 match_threshold: float = 0.82,
                 ambiguity_margin: float = 0.05):
        self.match_threshold = match_threshold
        # If the gap between the top match and second distinct person match is less than this margin,
        # it is considered ambiguous.
        self.ambiguity_margin = ambiguity_margin

    def resolve(self, search_results: List[Tuple[str, float]]) -> Tuple[ResolutionStatus, str]:
        """
        Takes a list of (canonical_person_id, similarity_score) tuples.
        Returns the resolution status and the canonical_person_id if matched.
        """
        if not search_results:
            return ResolutionStatus.NEW, None

        # Deduplicate by canonical_person_id, retaining the highest score per distinct person
        person_best_scores = {}
        for pid, score in search_results:
            if pid not in person_best_scores or score > person_best_scores[pid]:
                person_best_scores[pid] = score

        distinct_sorted = sorted(person_best_scores.items(), key=lambda x: x[1], reverse=True)

        top_match_id, top_score = distinct_sorted[0]

        if top_score < self.match_threshold:
            # Not similar enough to any known person
            return ResolutionStatus.NEW, None

        if len(distinct_sorted) > 1:
            second_match_id, second_score = distinct_sorted[1]
            # Check for ambiguity: Is the second best distinct person too close in similarity?
            if (top_score - second_score) < self.ambiguity_margin:
                return ResolutionStatus.UNRESOLVED, None

        return ResolutionStatus.MATCHED, top_match_id
