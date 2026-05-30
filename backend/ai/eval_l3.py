import sys
sys.path.insert(0, '.')
from ai.train_env import TrainDispatchEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from collections import Counter

def mask_fn(env): return env.get_action_mask()

def make_env():
    e = TrainDispatchEnv()
    e.set_difficulty(7)
    return ActionMasker(e, mask_fn)

# Load Level 3 best model
raw = make_vec_env(make_env, n_envs=1)
model = MaskablePPO.load("ai/models/L3_7Trains_Best/best_model.zip")

# Run 20 episodes, collect diagnostics
episode_lengths = []
episode_rewards = []
timeout_count   = 0
collision_count = 0
deadlock_snapshots = []

for ep in range(20):
    env = TrainDispatchEnv()
    env.set_difficulty(7)
    env = ActionMasker(env, mask_fn)
    obs, _ = env.reset()
    
    ep_reward = 0
    step = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Handle both 4-tuple and 5-tuple step returns
        step_return = env.step(action)
        if len(step_return) == 5:
            obs, rew, terminated, truncated, info = step_return
            done = terminated or truncated
        else:
            obs, rew, done, info = step_return
        ep_reward += rew
        step += 1
        
        # Sample train states every 100 steps
        if step % 100 == 0:
            inner = env.env
            positions  = [t['position'] for t in inner.trains]
            speeds     = [t['speed'] for t in inner.trains]
            directions = [t['direction'] for t in inner.trains]
            holding    = sum(1 for s in speeds if s == 0)
            finished   = sum(1 for t in inner.trains if t['finished'])
            token_status = inner.ghat_token.status()
            
            if holding >= 4:
                deadlock_snapshots.append({
                    'ep': ep, 'step': step,
                    'positions': positions,
                    'speeds': speeds,
                    'directions': directions,
                    'holding': holding,
                    'finished': finished,
                    'token': token_status,
                })
    
    episode_lengths.append(step)
    episode_rewards.append(ep_reward)
    if step >= 1490:
        timeout_count += 1

print(f"\n{'='*55}")
print(f"LEVEL 3 DIAGNOSTIC — 20 Episodes")
print(f"{'='*55}")
print(f"Avg episode length : {np.mean(episode_lengths):.0f} steps")
print(f"Avg reward         : {np.mean(episode_rewards):.0f}")
print(f"Timeouts           : {timeout_count}/20")
print(f"Deadlock snapshots : {len(deadlock_snapshots)}")

if deadlock_snapshots:
    print(f"\nSample deadlock at ep={deadlock_snapshots[0]['ep']} "
          f"step={deadlock_snapshots[0]['step']}:")
    s = deadlock_snapshots[0]
    print(f"  Trains holding : {s['holding']}/7")
    print(f"  Finished       : {s['finished']}/7")
    print(f"  Token status   : {s['token']}")
    print(f"  Directions     : {s['directions']}")
    print(f"  Positions      : {s['positions']}")
print(f"{'='*55}")
