import sys
sys.path.insert(0, '.')
from ai.train_env import TrainDispatchEnv
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

def mask_fn(env): return env.get_action_mask()

env = TrainDispatchEnv()
env.set_difficulty(7)
obs, _ = env.reset()

# Fast forward to step 15 (after first train's start_time=14)
acts = np.zeros(10, dtype=int)
for _ in range(15):
    step_return = env.step(acts)
    if len(step_return) == 5:
        obs, rew, terminated, truncated, info = step_return
    else:
        obs, rew, done, info = step_return

mask = env.get_action_mask()
print(f"sim_time={env.sim_time}")
for i, train in enumerate(env.trains):
    sched = env.schedule[train['id']]
    print(f"  Train {i}: pos={train['position']:>4} "
          f"start={sched['start_time']:>3} "
          f"PROCEED={mask[i,1]} "
          f"({'ready' if env.sim_time >= sched['start_time'] else 'waiting'})")
