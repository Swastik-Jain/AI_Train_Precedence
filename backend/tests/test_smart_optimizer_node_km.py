from or_tools.smart_optimizer import SmartOptimizer

def test_resolve_main_target_prefers_lower_km_for_up_trains():
    """
    Regression test: _resolve_main_target's direction-aware logic was fully
    implemented but never received real node_km data (the caller passed
    node_km={} implicitly), so it always tied at 0 and fell back to
    next_opts[0] regardless of direction. This directly tests the resolver
    with real km data and confirms an UP train prefers the lower-km option.
    """
    opt = SmartOptimizer()
    node_km = {10: 50.0, 20: 45.0, 30: 55.0}  # node 20 is closer to CSMT (lower km)
    target = opt._resolve_main_target(
        pos=10, next_opts=[30, 20], direction='UP', node_km=node_km, track_map={}
    )
    assert target == 20, "UP train should prefer the lower-km branch, not just next_opts[0]"

def test_resolve_main_target_falls_back_when_node_km_missing():
    """With no km data (the old broken behavior), confirm it degrades to next_opts[0]
    predictably rather than crashing — this is the fallback path, not the fix itself."""
    opt = SmartOptimizer()
    target = opt._resolve_main_target(
        pos=10, next_opts=[30, 20], direction='UP', node_km={}, track_map={}
    )
    assert target == 30  # next_opts[0] — documents the pre-fix fallback behavior explicitly
