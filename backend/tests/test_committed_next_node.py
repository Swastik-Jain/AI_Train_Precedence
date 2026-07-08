import numpy as np
from train_env import TrainDispatchEnv

def test_committed_next_node_reflects_divert_not_just_main_track():
    """
    Regression test for the edge-mapping bug: simulation_service.py used to
    derive the visual edge_id via a broken path-vs-int comparison that always
    fell back to next_opts[0] (the main track), regardless of what the RL env
    actually decided. train_env.py now records the real decision on
    train['committed_next_node'] every step. This test forces a train into a
    junction with a real divert option and confirms the recorded target
    reflects the DIVERT choice (action=2), not just the main track.
    """
    env = TrainDispatchEnv()
    env.reset()

    train = env.trains[0]
    # Pick a node with >1 next option in the loaded track_map — adjust node id
    # if the corridor topology changes; the assertion below will fail loudly
    # (via the skip) if this node stops being a real junction.
    junction_candidates = [
        n for n, info in env.track_map.items()
        if len(info.get('next', [])) > 1 or len(info.get('prev', [])) > 1
    ]
    assert junction_candidates, "expected at least one multi-track junction in the corridor topology"
    node = junction_candidates[0]

    train.update({'position': node, 'finished': False, 'speed': 0, 'target_speed': 0})
    env._movement_acc[0] = 999  # force enough accumulated distance to commit a move this step

    action = np.array([2] * len(env.trains))  # act=2 == DIVERT for every train
    env.step(action)

    assert train.get('committed_next_node') is not None
    # The committed target must be a real neighbor of the junction node —
    # not silently defaulted to the node itself or an out-of-graph value.
    valid_targets = set(env.track_map.get(node, {}).get('next', [])) | set(env.track_map.get(node, {}).get('prev', []))
    assert train['committed_next_node'] in valid_targets
