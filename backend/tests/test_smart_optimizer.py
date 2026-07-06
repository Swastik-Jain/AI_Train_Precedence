import copy
import numpy as np
from train_env import TrainDispatchEnv
from or_tools.smart_optimizer import SmartOptimizer


def test_banker_attached_trains_get_a_safe_action():
    """
    Regression test for the banker-deadlock scenario: three UP trains sharing
    two occupied nodes near the ghat, all banker-attached and stationary.
    The shield must return a valid action for every train — none should be
    silently dropped or left without a safe fallback.
    """
    env = TrainDispatchEnv()
    env.reset()
    while len(env.trains) < 3:
        env.trains.append(copy.deepcopy(env.trains[0]))

    env.trains[0].update({'position': 49, 'direction': 'UP', 'banker_wait': 0,
                           'banker_attached': True, 'speed': 0, 'target_speed': 0,
                           'finished': False, 'priority': 10, 'id': 'SUP_101'})
    env.trains[1].update({'position': 1033, 'direction': 'UP', 'banker_wait': 0,
                           'banker_attached': True, 'speed': 0, 'target_speed': 0,
                           'finished': False, 'priority': 10, 'id': 'GOO_102'})
    env.trains[2].update({'position': 50, 'direction': 'UP', 'banker_wait': 0,
                           'banker_attached': True, 'speed': 0, 'target_speed': 0,
                           'finished': False, 'priority': 10, 'id': 'SUP_103'})
    env._occupancy = {49: {'UP': 1}, 1033: {'UP': 1}, 50: {'UP': 1}}

    opt = SmartOptimizer()
    safe_actions, _ = opt.optimize_decision(
        env.trains, np.array([1, 1, 1]), env.track_map, env.ghat_token
    )

    assert len(safe_actions) == 3, "shield dropped a train instead of returning an action for each"
    assert all(a in (0, 1, 2) for a in safe_actions), f"got an out-of-range action: {safe_actions}"
