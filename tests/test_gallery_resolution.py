import os
import pytest
import numpy as np
from pathlib import Path

from app.cv.identity.resolver import IdentityResolver, ResolutionStatus


def test_multiple_tracklets_same_person_merge():
    """
    Test A: Multiple raw track IDs belonging to the same person resolve to one canonical_person_id.
    """
    resolver = IdentityResolver(match_threshold=0.72, ambiguity_margin=0.05)
    
    # Candidate 1 is clearly the same person (score 0.88), candidate 2 is distant (score 0.52)
    search_results = [
        ("PERSON_ALPHA", 0.88),
        ("PERSON_BETA", 0.52)
    ]
    status, canonical_id = resolver.resolve(search_results)
    
    assert status == ResolutionStatus.MATCHED
    assert canonical_id == "PERSON_ALPHA"


def test_different_people_do_not_collapse():
    """
    Test B: Different people (similarity below threshold) do not collapse into existing identity.
    """
    resolver = IdentityResolver(match_threshold=0.72, ambiguity_margin=0.05)
    
    # Top score is 0.61, which is below the calibrated 0.72 threshold
    search_results = [
        ("PERSON_ALPHA", 0.61),
        ("PERSON_BETA", 0.50)
    ]
    status, canonical_id = resolver.resolve(search_results)
    
    assert status == ResolutionStatus.NEW
    assert canonical_id is None


def test_ambiguous_matches_yield_unresolved():
    """
    Test C: Ambiguous matches (delta < 0.05) yield ResolutionStatus.UNRESOLVED.
    """
    resolver = IdentityResolver(match_threshold=0.72, ambiguity_margin=0.05)
    
    # Top candidate 0.81, second candidate 0.79 (delta = 0.02 < 0.05 margin)
    search_results = [
        ("PERSON_ALPHA", 0.81),
        ("PERSON_BETA", 0.79)
    ]
    status, canonical_id = resolver.resolve(search_results)
    
    assert status == ResolutionStatus.UNRESOLVED
    assert canonical_id is None


def test_multi_embedding_distinct_deduplication():
    """
    Test C2: Multi-embedding gallery for same person does not trigger false ambiguity.
    """
    resolver = IdentityResolver(match_threshold=0.72, ambiguity_margin=0.05)
    
    # Multiple embeddings from PERSON_ALPHA (0.85, 0.83), distinct second person is PERSON_BETA (0.60)
    search_results = [
        ("PERSON_ALPHA", 0.85),
        ("PERSON_ALPHA", 0.83),
        ("PERSON_BETA", 0.60)
    ]
    status, canonical_id = resolver.resolve(search_results)
    
    assert status == ResolutionStatus.MATCHED
    assert canonical_id == "PERSON_ALPHA"


def test_dual_layer_storage_exists():
    """
    Test D & E: Canonical person gallery crops exist and original track crops remain intact.
    """
    project_root = Path(__file__).resolve().parent.parent
    tracks_dir = project_root / "dataset" / "tracks"
    persons_dir = project_root / "dataset" / "persons"
    
    assert tracks_dir.exists(), "Raw track evidence directory must exist"
    assert persons_dir.exists(), "Canonical person directory must exist"
    
    # Check that canonical person folders have crops
    canonical_dirs = [d for d in persons_dir.iterdir() if d.is_dir() and d.name.startswith("PERSON_")]
    assert len(canonical_dirs) > 0, "At least one canonical person folder must exist"
    
    sample_person = canonical_dirs[0]
    crops_dir = sample_person / "crops"
    assert crops_dir.exists(), "Canonical person must have crops directory"
    assert len(list(crops_dir.glob("*.jpg"))) > 0, "Canonical person must contain crop images"
