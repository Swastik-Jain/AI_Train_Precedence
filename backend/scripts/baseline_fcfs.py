"""
baseline_fcfs.py — First-Come-First-Served Baseline
Trains are processed in order of their start_time.
No priority, no anticipation, no optimization.
The simplest possible dispatcher.

Fix applied (2026-05-21):
    Delay calculation now uses each train's actual finish_step instead
    of episode end time (ep_len). Unfinished trains are counted as
    max-delayed using ep_len. avg_delay denominator changed from
    len(finished) to len(all trains) so it penalises non-completion.
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_env import TrainDispatchEnv
from ai.config import MAX_TRAINS_CAPACITY


def fcfs_action(trains, track_map, schedule, node_km=None):
    """
    Generate actions using first-come-first-served ordering.
    Trains that entered the section earlier get priority.
    """
    actions = np.zeros(MAX_TRAINS_CAPACITY, dtype=int)
    node_km = node_km or {}

    occupancy = defaultdict(int)
    for t in trains:
        p = t['position']
        if p not in (0, 998, 999):
            occupancy[p] += 1

    claimed = defaultdict(int)

    # Sort by start_time ascending — earliest spawn gets priority
    sorted_idx = sorted(
        range(len(trains)),
        key=lambda k: schedule.get(
            trains[k]['id'], {}
        ).get('start_time', 999),
    )

    for i in sorted_idx:
        train = trains[i]
        pos   = train['position']

        if train['finished'] or pos == 999:
            actions[i] = 0
            continue
        if pos in (0, 998):
            actions[i] = 0
            continue
        if train.get('dwell_rem', 0) > 0:
            actions[i] = 0
            continue
        if train.get('banker_wait', 0) > 0:
            actions[i] = 0
            continue

        node_data   = track_map.get(pos, {})
        next_opts   = node_data.get('next', [])
        direction   = train.get('direction', 'DOWN')

        if not next_opts:
            actions[i] = 0
            continue

        if direction == 'UP' and node_km:
            my_km      = node_km.get(pos, 0)
            candidates = [n for n in next_opts if node_km.get(n, 0) <= my_km]
            main_target = (
                min(candidates, key=lambda n: node_km.get(n, 0))
                if candidates else next_opts[0]
            )
        else:
            main_target = next_opts[0]

        loop_targets = [n for n in next_opts if n != main_target]
        main_cap     = track_map.get(main_target, {}).get('capacity', 1)
        main_occ     = occupancy[main_target] + claimed[main_target]

        if main_occ < main_cap:
            actions[i] = 1
            claimed[main_target] += 1
        else:
            loop_found = False
            for ln in loop_targets:
                ln_cap = track_map.get(ln, {}).get('capacity', 1)
                ln_occ = occupancy[ln] + claimed[ln]
                if ln_occ < ln_cap:
                    actions[i] = 2
                    claimed[ln] += 1
                    loop_found = True
                    break
            if not loop_found:
                actions[i] = 0
                claimed[pos] += 1

    return actions


def run_fcfs(
    num_trains=10,
    num_episodes=200,
    seed=42,
    save_path=None,
    verbose=True,
):
    print(f"\n{'='*60}")
    print(f"FCFS BASELINE | {num_trains} trains | {num_episodes} episodes")
    print(f"{'='*60}")

    rewards    = []
    lengths    = []
    timeouts   = []
    collisions = []
    on_time    = []
    delays     = []
    ghat_waits = []
    hp_delays  = []
    efficiencies = []

    for ep in range(num_episodes):
        env = TrainDispatchEnv()
        env.set_difficulty(num_trains)
        obs, _ = env.reset(seed=seed + ep)

        ep_reward    = 0.0
        ep_len       = 0
        ep_collision = False
        done         = False
        ghat_wait_ep = 0

        node_km = {
            nid: data.get('km', 0)
            for nid, data in env.track_map.items()
        }

        while not done:
            action = fcfs_action(
                trains=env.trains,
                track_map=env.track_map,
                schedule=env.schedule,
                node_km=node_km,
            )
            obs, rew, done, _, _ = env.step(action)
            ep_reward += rew
            ep_len    += 1
            if rew <= -70:
                ep_collision = True
                
            occupied_by_node = {
                t['position']: {'train_id': t['id'], 'direction': t.get('direction', 'DOWN')}
                for t in env.trains if not t['finished']
            }
            ksr_q = env.ghat_token.compute_queue(env.track_map, occupied_by_node, 'KSR')
            igp_q = env.ghat_token.compute_queue(env.track_map, occupied_by_node, 'IGP')
            ghat_wait_ep += len(ksr_q) + len(igp_q)

        finished    = [t for t in env.trains if t['finished']]
        unfinished  = [t for t in env.trains if not t['finished']]
        total       = len(env.trains)
        on_time_cnt = 0
        total_delay = 0
        hp_delay_ep = 0
        hp_count = 0

        for t in finished:
            sched    = env.schedule.get(t['id'], {})
            deadline = sched.get('deadline', 0)
            # Use actual finish step, not episode end time.
            actual_finish = t.get('finish_step') if t.get('finish_step') is not None else ep_len
            delay    = max(0, actual_finish - deadline)
            total_delay += delay
            if delay == 0:
                on_time_cnt += 1
            if t.get('priority', 0) >= 5:
                hp_delay_ep += delay
                hp_count += 1

        # Unfinished trains count as max-delayed
        for t in unfinished:
            sched    = env.schedule.get(t['id'], {})
            deadline = sched.get('deadline', 0)
            delay    = max(0, ep_len - deadline)
            total_delay += delay
            if t.get('priority', 0) >= 5:
                hp_delay_ep += delay
                hp_count += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)
        timeouts.append(int(ep_len >= 1490))
        collisions.append(int(ep_collision))
        on_time.append(on_time_cnt / max(total, 1) * 100)
        delays.append(total_delay / max(total, 1))
        ghat_waits.append(ghat_wait_ep)
        hp_delays.append(hp_delay_ep / max(hp_count, 1))
        efficiencies.append(len(finished) / max(ep_len, 1))

        if verbose and ep % 50 == 0:
            print(f"  Ep {ep:>3} | reward={ep_reward:>8.1f} | "
                  f"len={ep_len:>4} | on_time={on_time[-1]:.1f}%")

    summary = {
        'baseline':        'fcfs',
        'num_trains':      num_trains,
        'num_episodes':    num_episodes,
        'avg_reward':      round(float(np.mean(rewards)), 2),
        'std_reward':      round(float(np.std(rewards)), 2),
        'avg_ep_length':   round(float(np.mean(lengths)), 1),
        'timeout_rate':    round(float(np.mean(timeouts)) * 100, 1),
        'collision_rate':  round(float(np.mean(collisions)) * 100, 1),
        'avg_on_time_pct': round(float(np.mean(on_time)), 1),
        'avg_delay':       round(float(np.mean(delays)), 1),
        'avg_ghat_wait':   round(float(np.mean(ghat_waits)), 1),
        'avg_hp_delay':    round(float(np.mean(hp_delays)), 1),
        'avg_efficiency':  round(float(np.mean(efficiencies)), 3),
    }

    print(f"\nFCFS RESULTS:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {save_path}")

    return summary


if __name__ == '__main__':
    run_fcfs(
        num_trains=10,
        num_episodes=200,
        seed=42,
        save_path='results/fcfs.json',
    )
