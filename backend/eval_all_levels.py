"""
eval_all_levels.py — Full curriculum evaluation across all 6 Phase-4 levels.
Run from backend/: python eval_all_levels.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from train_env import TrainDispatchEnv

def mask_fn(env): return env.get_action_mask()

LEVELS = [
    {"name": "L1 –  2 Trains", "difficulty": 2,  "model": "ai/models/Phase3/L1_2Trains_Best/best_model"},
    {"name": "L2 –  5 Trains", "difficulty": 5,  "model": "ai/models/Phase3/L2_5Trains_Best/best_model"},
    {"name": "L3 –  7 Trains", "difficulty": 7,  "model": "ai/models/Phase3/L3_7Trains_Best/best_model"},
    {"name": "L4 – 10 Trains", "difficulty": 10, "model": "ai/models/Phase3/L4_10Trains_Best/best_model"},
    {"name": "L5 – 15 Trains", "difficulty": 15, "model": "ai/models/Phase3/L5_15Trains_Best_v2/best_model"},
    {"name": "L6 – 25 Trains", "difficulty": 25, "model": "ai/models/Phase3/L6_25Trains_Best_v2/best_model"},
]

# Phase-4 pass criteria (post reward-normalisation)
PASS_CRITERIA = {
    2:  {"reward": 8.0,  "completion": 1.00},
    5:  {"reward": 20.0, "completion": 0.95},
    7:  {"reward": 28.0, "completion": 0.90},
    10: {"reward": 15.0, "completion": 0.80},
    15: {"reward": 5.0,  "completion": 0.55},
    25: {"reward": 0.0,  "completion": 0.35},
}

N_EPISODES = 30
MAX_STEPS  = 1490

print("\n" + "=" * 70)
print("  PHASE-4 CURRICULUM EVALUATION — All Levels (30 episodes each)")
print("=" * 70)

summary_rows = []

for lvl in LEVELS:
    diff = lvl["difficulty"]
    print(f"\n▶  {lvl['name']}")
    try:
        model = MaskablePPO.load(lvl["model"])
    except FileNotFoundError as e:
        print(f"   ❌  Model not found: {e}")
        continue

    rewards, lengths, timeouts, completions, deadlocks = [], [], 0, [], 0

    for ep in range(N_EPISODES):
        env = TrainDispatchEnv()
        env.set_difficulty(diff)
        env = ActionMasker(env, mask_fn)
        obs, _ = env.reset()

        done, ep_r, ep_l, ep_deadlock = False, 0.0, 0, False

        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(act)
            ep_r += r
            ep_l += 1
            # Heuristic: if reward suddenly drops a lot, likely a deadlock/collision
            if r < -50:
                ep_deadlock = True

        rewards.append(ep_r)
        lengths.append(ep_l)
        if ep_l >= MAX_STEPS:
            timeouts += 1
        if ep_deadlock:
            deadlocks += 1

        # Count finished trains
        inner = env.env if hasattr(env, 'env') else env
        if hasattr(inner, 'trains'):
            fin = sum(1 for t in inner.trains if t.get('finished', False))
            completions.append(fin / diff)
        else:
            completions.append(0.0)

    mean_r   = np.mean(rewards)
    std_r    = np.std(rewards)
    mean_l   = np.mean(lengths)
    comp_pct = np.mean(completions) * 100

    crit = PASS_CRITERIA.get(diff, {"reward": 0, "completion": 0})
    reward_ok  = mean_r  >= crit["reward"]
    comp_ok    = (comp_pct / 100) >= crit["completion"]

    if reward_ok and comp_ok:
        grade = "✅ PASS"
    elif mean_r > -200:
        grade = "⚠️  PARTIAL"
    else:
        grade = "❌ FAIL"

    print(f"   Avg Reward   : {mean_r:>9.1f}  (±{std_r:.1f})")
    print(f"   Avg Length   : {mean_l:>9.1f} steps")
    print(f"   Timeouts     : {timeouts:>2}/{N_EPISODES}")
    print(f"   Deadlocks    : {deadlocks:>2}/{N_EPISODES}")
    print(f"   Completion % : {comp_pct:>8.1f}%  (target: {crit['completion']*100:.0f}%)")
    print(f"   Grade        : {grade}")

    summary_rows.append({
        "name": lvl["name"], "reward": mean_r, "std": std_r,
        "length": mean_l, "timeouts": timeouts, "deadlocks": deadlocks,
        "completion": comp_pct, "grade": grade,
    })

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  SUMMARY TABLE")
print("=" * 70)
hdr = f"{'Level':<18} | {'AvgRew':>8} | {'Std':>6} | {'AvgLen':>7} | {'Comp%':>6} | {'TO':>3} | {'DL':>3} | Grade"
print(hdr)
print("-" * 70)
for r in summary_rows:
    print(f"{r['name']:<18} | {r['reward']:>8.1f} | {r['std']:>6.1f} | "
          f"{r['length']:>7.1f} | {r['completion']:>5.1f}% | "
          f"{r['timeouts']:>3} | {r['deadlocks']:>3} | {r['grade']}")
print("=" * 70 + "\n")
