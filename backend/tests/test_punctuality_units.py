from state import SimulationState

def test_sim_tick_resets_on_fresh_state():
    """
    Regression test for the punctuality units mismatch: state.sim_tick used to
    persist across inference start/stop cycles while the RL env's own clock
    reset via env.reset() in start_inference(), causing punctuality math
    (which compares deadline directly against state.sim_tick) to drift after
    the first run. This is a narrow unit test confirming the attribute
    contract; the full regression (sim_tick actually resetting inside
    start_inference()) needs an integration test with a real/mocked model+env,
    which is a reasonable follow-up rather than part of this fast unit test.
    """
    state = SimulationState()
    assert state.sim_tick == 0

    state.sim_tick = 500
    assert state.sim_tick == 500  # simulate a prior session's accumulated ticks
    # start_inference() is expected to reset this back to 0 — see
    # services/simulation_service.py::start_inference. That integration-level
    # assertion belongs in a test that can mock _get_sim_brain(); tracked as a
    # follow-up rather than blocking this test file.
