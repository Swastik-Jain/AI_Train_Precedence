import sys
sys.path.insert(0, '.')
import random
import numpy as np
from train_env import TrainDispatchEnv

N_EPISODES = 20
MAX_STEPS = 1490
random.seed(0)
np.random.seed(0)

violations = []

for ep in range(N_EPISODES):
    env = TrainDispatchEnv()
    env.set_difficulty(15)
    obs, _ = env.reset()

    for step in range(MAX_STEPS):
        mask = env.get_action_mask()
        n_trains = len(env.trains)
        actions = []
        for i in range(mask.shape[0]):
            valid = np.where(mask[i])[0]
            if len(valid) == 0:
                actions.append(0)
            else:
                actions.append(int(random.choice(valid)))
        action = np.array(actions)

        obs, reward, terminated, truncated, info = env.step(action)

        # Check invariant: total occupancy per node must never exceed capacity
        for node_id, occ_by_dir in env._occupancy.items():
            total = sum(occ_by_dir.values())
            cap = env.track_map.get(node_id, {}).get('capacity', 1)
            if total > cap:
                node_type = env.track_map.get(node_id, {}).get('type')
                violations.append({
                    'episode': ep, 'step': step, 'node': node_id,
                    'type': node_type, 'capacity': cap,
                    'occupancy_by_direction': dict(occ_by_dir),
                    'total': total,
                })

        if terminated or truncated:
            break

    print(f"Episode {ep}: done at step {step}, violations so far: {len(violations)}")

print("\n" + "=" * 60)
if violations:
    print(f"FOUND {len(violations)} CAPACITY VIOLATIONS:")
    for v in violations[:20]:
        print(v)
else:
    print("NO capacity violations found across", N_EPISODES, "episodes of random valid actions.")
print("=" * 60)
