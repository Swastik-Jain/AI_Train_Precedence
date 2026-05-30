import sys
sys.path.insert(0, '.')
from ai.train_env import TrainDispatchEnv
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

def mask_fn(env): return env.get_action_mask()

env = TrainDispatchEnv()
env.set_difficulty(7)
obs, _ = env.reset()

print("=== SPAWN STATE ===")
for i, train in enumerate(env.trains):
    pos       = train['position']
    direction = train['direction']
    dwell     = train['dwell_rem']
    banker_w  = train['banker_wait']
    finished  = train['finished']
    sched     = env.schedule[train['id']]
    start     = sched['start_time']
    
    mask      = env.get_action_mask()
    can_proc  = mask[i, 1]
    can_div   = mask[i, 2]
    
    node_data = env.track_map.get(pos, {})
    next_opts = node_data.get('next', [])
    
    print(f"\nTrain {i} | {train['id']}")
    print(f"  pos={pos} dir={direction} start_time={start} sim_time={env.sim_time}")
    print(f"  dwell={dwell} banker_wait={banker_w} finished={finished}")
    print(f"  node_type={node_data.get('type','?')} next={next_opts}")
    print(f"  mask: HOLD={mask[i,0]} PROCEED={can_proc} DIVERT={can_div}")

print(f"\n=== AFTER 20 STEPS (random HOLD) ===")
for step in range(20):
    acts = np.zeros(10, dtype=int)  # all HOLD
    step_return = env.step(acts)
    if len(step_return) == 5:
        obs, rew, terminated, truncated, info = step_return
        done = terminated or truncated
    else:
        obs, rew, done, info = step_return

mask = env.get_action_mask()
proceed_allowed = mask[:len(env.trains), 1].sum()
print(f"sim_time={env.sim_time}")
print(f"PROCEED allowed: {proceed_allowed}/{len(env.trains)}")
for i, train in enumerate(env.trains):
    print(f"  Train {i}: pos={train['position']} "
          f"start={env.schedule[train['id']]['start_time']} "
          f"PROCEED={mask[i,1]}")
