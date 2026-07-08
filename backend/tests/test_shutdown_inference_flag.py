from state import SimulationState

def test_shutdown_flag_lives_on_state_not_globals():
    """
    Regression test: _SHUTDOWN_INFERENCE_FLAG used to be a bare module-global
    inside simulate_trains_bg, silently scoped as a local variable (no `global`
    declaration), so the auto-stop-after-episode logic never actually fired.
    Confirms the flag now lives on SimulationState, where simulate_trains_bg's
    tick-loop tail can actually read what the episode-termination branch wrote.
    """
    state = SimulationState()
    assert hasattr(state, "shutdown_inference_flag")
    assert state.shutdown_inference_flag is False

    state.shutdown_inference_flag = True
    assert state.shutdown_inference_flag is True
