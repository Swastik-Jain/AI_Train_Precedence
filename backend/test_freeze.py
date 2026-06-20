import sys
sys.path.append('.')
from train_env import TrainDispatchEnv
import numpy as np
import copy

env = TrainDispatchEnv()
env.reset()

while len(env.trains) < 3:
    env.trains.append(copy.deepcopy(env.trains[0]))

env.trains[0].update({'position': 49, 'direction': 'UP', 'banker_wait': 0, 'banker_attached': True, 'speed': 0, 'target_speed': 0, 'finished': False, 'priority': 10, 'id': 'SUP_101'})
env.trains[1].update({'position': 1033, 'direction': 'UP', 'banker_wait': 0, 'banker_attached': True, 'speed': 0, 'target_speed': 0, 'finished': False, 'priority': 10, 'id': 'GOO_102'})
env.trains[2].update({'position': 50, 'direction': 'UP', 'banker_wait': 0, 'banker_attached': True, 'speed': 0, 'target_speed': 0, 'finished': False, 'priority': 10, 'id': 'SUP_103'})

env._occupancy = {49: {'UP': 1}, 1033: {'UP': 1}, 50: {'UP': 1}}

from or_tools.smart_optimizer import SmartOptimizer
opt = SmartOptimizer()
safe_actions, _ = opt.optimize_decision(env.trains, np.array([1, 1, 1]), env.track_map, env.ghat_token)
print(f"Safe Actions: {safe_actions}")
