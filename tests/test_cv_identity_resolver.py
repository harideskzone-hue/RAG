from app.cv.identity.resolver import IdentityResolver, ResolutionStatus

def test_identity_resolver_matched():
    resolver = IdentityResolver(match_threshold=0.85, ambiguity_margin=0.05)
    
    search_results = [
        ("P001", 0.92),
        ("P002", 0.70)
    ]
    status, pid = resolver.resolve(search_results)
    assert status == ResolutionStatus.MATCHED
    assert pid == "P001"

def test_identity_resolver_new():
    resolver = IdentityResolver(match_threshold=0.85, ambiguity_margin=0.05)
    
    # Highest score is below threshold
    search_results = [
        ("P001", 0.80),
        ("P002", 0.70)
    ]
    status, pid = resolver.resolve(search_results)
    assert status == ResolutionStatus.NEW
    assert pid is None
    
    # Empty results
    status, pid = resolver.resolve([])
    assert status == ResolutionStatus.NEW
    assert pid is None

def test_identity_resolver_unresolved():
    resolver = IdentityResolver(match_threshold=0.85, ambiguity_margin=0.05)
    
    # Highest score is above threshold, but the second highest is too close (< 0.05 difference)
    search_results = [
        ("P001", 0.90),
        ("P002", 0.88)
    ]
    status, pid = resolver.resolve(search_results)
    assert status == ResolutionStatus.UNRESOLVED
    assert pid is None
