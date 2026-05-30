import sys
sys.path.insert(0, '.')
from ai.train_env import TrainDispatchEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
import numpy as np

def mask_fn(env): return env.get_action_mask()

env = TrainDispatchEnv()
env.set_difficulty(7)
env = ActionMasker(env, mask_fn)
model = MaskablePPO.load("ai/models/L3_7Trains_Best/best_model.zip")

obs, _ = env.reset()
action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, PROCEED, DIVERT

for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    for a in action[:7]: # count only active trains
        action_counts[int(a)] += 1
        
    step_return = env.step(action)
    if len(step_return) == 5:
        obs, rew, terminated, truncated, info = step_return
        done = terminated or truncated
    else:
        obs, rew, done, info = step_return
        
    if done:
        break

total = sum(action_counts.values())
print(f"Action distribution over {step+1} steps x 7 trains:")
if total > 0:
    print(f"  HOLD   : {action_counts[0]/total*100:.1f}%")
    print(f"  PROCEED: {action_counts[1]/total*100:.1f}%")
    print(f"  DIVERT : {action_counts[2]/total*100:.1f}%")
print()

# Also check what mask allows
obs, _ = env.reset()
mask = env.action_masks()
proceed_allowed = mask[:7, 1].sum()
divert_allowed  = mask[:7, 2].sum()
print(f"At reset — mask allows PROCEED for {proceed_allowed}/7 trains")
print(f"At reset — mask allows DIVERT  for {divert_allowed}/7 trains")
