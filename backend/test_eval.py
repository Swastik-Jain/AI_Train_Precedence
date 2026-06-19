import sys
sys.path.insert(0, '.')
from train_env import TrainDispatchEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
import numpy as np

def mask_fn(env): return env.get_action_mask()

model = MaskablePPO.load("ai/models/Phase3/L4_10Trains_Best/best_model")
rewards, lengths = [], []
for ep in range(30):
    env = TrainDispatchEnv()
    env.set_difficulty(10)
    env = ActionMasker(env, mask_fn)
    obs, _ = env.reset()
    done = False
    ep_r, ep_l = 0, 0
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(act)
        ep_r += r
        ep_l += 1
    rewards.append(ep_r)
    lengths.append(ep_l)

print(f"Avg reward : {np.mean(rewards):.1f}")
print(f"Avg length : {np.mean(lengths):.1f}")
print(f"Timeouts   : {sum(1 for l in lengths if l >= 1490)}/30")
