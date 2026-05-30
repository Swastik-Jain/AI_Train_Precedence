"""
benchmark.py — Full Benchmark Table Generator (v3)
Runs all four dispatchers on identical random seeds and produces
the comparison table that is the project's primary deliverable.

v3 changes (2026-05-21):
  - Per-train finish_step delay tracking: delay now uses each train's
    actual completion step, not episode end time. This was the primary
    cause of all dispatchers showing near-identical metrics.
  - Unfinished trains counted as max-delayed (ep_len), not dropped.
  - SmartOptimizer anti-loitering fix now in smart_optimizer.py.

v2 changes:
  - Stress-test mode (clustered spawns + 50% deadlines)
  - Per-train finish-step delay tracking (not episode-level)
  - Worst-case resilience metrics (min, p5, std)
  - Ghat waiting time, high-priority delay, efficiency

Four dispatchers:
    1. Rule-based    (priority order — what IR does today)
    2. FCFS          (first-come-first-served)
    3. Your RL Agent (trained MaskablePPO)
    4. RL + Shield   (RL with CP-SAT feasibility shield)

Usage:
    # Normal evaluation
    python benchmark.py --model ai/models/ppo_L4_10Trains_final.zip

    # Stress test (recommended — shows real differences)
    python benchmark.py --model ai/models/ppo_L4_10Trains_final.zip --stress
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_env import TrainDispatchEnv
from ai.config import MAX_TRAINS_CAPACITY
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

try:
    from or_tools.feasibility_shield import FeasibilityShield
    from or_tools.smart_optimizer    import SmartOptimizer
    SHIELD_AVAILABLE = True
except ImportError:
    SHIELD_AVAILABLE = False
    print("⚠️  or_tools not found — skipping shield baseline")


def mask_fn(env):
    return env.get_action_mask()


# ─────────────────────────────────────────────────────────────────────────────
# Common episode runner — used by ALL dispatchers
# ─────────────────────────────────────────────────────────────────────────────

def _get_ghat_nodes(env):
    """Extract all node IDs near Kasara/Igatpuri for ghat wait tracking."""
    kasara_data = env.station_nodes.get('KASARA', {})
    igatpuri_data = env.station_nodes.get('IGATPURI', {})
    nodes = kasara_data.get('platforms', []) + igatpuri_data.get('platforms', [])
    for key in ['switch_in', 'switch_out']:
        if key in kasara_data:
            nodes.append(kasara_data[key])
        if key in igatpuri_data:
            nodes.append(igatpuri_data[key])
    return set(nodes)


def _collect_episode_metrics(env, ep_len):
    """
    Collect per-train metrics at end of episode.

    Uses each train's actual finish_step for delay calculation.
    Previously used ep_len for all trains which inflated delays for
    early finishers and washed out real dispatcher-quality differences.
    Trains that did NOT finish (timed-out) fall back to ep_len so they
    are still counted as late rather than silently dropped.
    """
    finished = [t for t in env.trains if t['finished']]
    total = len(env.trains)

    on_time_cnt = 0
    total_delay = 0
    hp_delay = 0
    hp_count = 0

    for t in finished:
        sched = env.schedule.get(t['id'], {})
        deadline = sched.get('deadline', 0)
        # Use the train's actual finish step, not episode end time.
        # finish_step is None only if something went wrong; fall back to ep_len.
        actual_finish = t.get('finish_step') if t.get('finish_step') is not None else ep_len
        delay = max(0, actual_finish - deadline)
        total_delay += delay
        if delay == 0:
            on_time_cnt += 1
        if t.get('priority', 0) >= 5:
            hp_delay += delay
            hp_count += 1

    # Unfinished trains count as maximum-delayed (ep_len used as finish time)
    unfinished = [t for t in env.trains if not t['finished']]
    for t in unfinished:
        sched = env.schedule.get(t['id'], {})
        deadline = sched.get('deadline', 0)
        delay = max(0, ep_len - deadline)
        total_delay += delay
        # unfinished trains are never on-time
        if t.get('priority', 0) >= 5:
            hp_delay += delay
            hp_count += 1

    all_trains = len(env.trains)
    return {
        'finished':    len(finished),
        'total':       all_trains,
        'on_time_pct': on_time_cnt / max(all_trains, 1) * 100,
        'avg_delay':   total_delay / max(all_trains, 1),
        'hp_delay':    hp_delay / max(hp_count, 1),
        'efficiency':  len(finished) / max(ep_len, 1),
    }


def _make_summary(label, num_trains, num_episodes, rewards, lengths,
                  timeouts, collisions, on_time, delays,
                  ghat_waits, hp_delays, efficiencies,
                  latencies=None):
    """Build the summary dict with worst-case resilience metrics."""
    summary = {
        'baseline':         label,
        'num_trains':       num_trains,
        'num_episodes':     num_episodes,
        'avg_reward':       round(float(np.mean(rewards)), 2),
        'std_reward':       round(float(np.std(rewards)), 2),
        'min_reward':       round(float(np.min(rewards)), 2),
        'p5_reward':        round(float(np.percentile(rewards, 5)), 2),
        'avg_ep_length':    round(float(np.mean(lengths)), 1),
        'max_ep_length':    round(float(np.max(lengths)), 0),
        'timeout_rate':     round(float(np.mean(timeouts)) * 100, 1),
        'collision_rate':   round(float(np.mean(collisions)) * 100, 1),
        'avg_on_time_pct':  round(float(np.mean(on_time)), 1),
        'avg_delay':        round(float(np.mean(delays)), 1),
        'avg_ghat_wait':    round(float(np.mean(ghat_waits)), 1),
        'avg_hp_delay':     round(float(np.mean(hp_delays)), 1),
        'avg_efficiency':   round(float(np.mean(efficiencies)), 4),
        'raw_rewards':      rewards,
    }
    if latencies:
        summary['avg_latency_ms'] = round(float(np.mean(latencies)), 3)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Rule-based dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_action(trains, track_map, node_km=None):
    """Priority-order rule-based dispatch (what IR does today)."""
    actions = np.zeros(MAX_TRAINS_CAPACITY, dtype=int)
    node_km = node_km or {}

    from collections import defaultdict
    occupancy = defaultdict(int)
    for t in trains:
        p = t['position']
        if p not in (0, 998, 999):
            occupancy[p] += 1

    claimed = defaultdict(int)
    sorted_idx = sorted(
        range(len(trains)),
        key=lambda k: trains[k].get('priority', 0),
        reverse=True,
    )

    for i in sorted_idx:
        train = trains[i]
        pos = train['position']

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

        node_data = track_map.get(pos, {})
        next_opts = node_data.get('next', [])
        direction = train.get('direction', 'DOWN')

        if not next_opts:
            actions[i] = 0
            continue

        if direction == 'UP' and node_km:
            my_km = node_km.get(pos, 0)
            candidates = [n for n in next_opts if node_km.get(n, 0) <= my_km]
            main_target = (
                min(candidates, key=lambda n: node_km.get(n, 0))
                if candidates else next_opts[0]
            )
        else:
            main_target = next_opts[0]

        loop_targets = [n for n in next_opts if n != main_target]
        main_cap = track_map.get(main_target, {}).get('capacity', 1)
        main_occ = occupancy[main_target] + claimed[main_target]

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


def fcfs_action(trains, track_map, schedule, node_km=None):
    """First-come-first-served dispatch."""
    actions = np.zeros(MAX_TRAINS_CAPACITY, dtype=int)
    node_km = node_km or {}

    from collections import defaultdict
    occupancy = defaultdict(int)
    for t in trains:
        p = t['position']
        if p not in (0, 998, 999):
            occupancy[p] += 1

    claimed = defaultdict(int)
    sorted_idx = sorted(
        range(len(trains)),
        key=lambda k: schedule.get(
            trains[k]['id'], {}
        ).get('start_time', 999),
    )

    for i in sorted_idx:
        train = trains[i]
        pos = train['position']

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

        node_data = track_map.get(pos, {})
        next_opts = node_data.get('next', [])
        direction = train.get('direction', 'DOWN')

        if not next_opts:
            actions[i] = 0
            continue

        if direction == 'UP' and node_km:
            my_km = node_km.get(pos, 0)
            candidates = [n for n in next_opts if node_km.get(n, 0) <= my_km]
            main_target = (
                min(candidates, key=lambda n: node_km.get(n, 0))
                if candidates else next_opts[0]
            )
        else:
            main_target = next_opts[0]

        loop_targets = [n for n in next_opts if n != main_target]
        main_cap = track_map.get(main_target, {}).get('capacity', 1)
        main_occ = occupancy[main_target] + claimed[main_target]

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


# ─────────────────────────────────────────────────────────────────────────────
# Unified runner for all baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_dispatcher(
    label,
    num_trains=10,
    num_episodes=200,
    seed=42,
    stress=False,
    chaos=False,
    hardcore=False,
    incident=False,
    dispatch_fn=None,
    model_path=None,
    use_shield=False,
    verbose=True,
):
    """
    Generic dispatcher evaluator.
    - dispatch_fn: callable(env) -> action array (for rule-based/FCFS)
    - model_path: path to MaskablePPO model (for RL agent)
    """
    print(f"\n{'='*60}")
    print(f"{label} | {num_trains} trains | {num_episodes} episodes"
          f"{' | STRESS' if stress else ''}"
          f"{' | CHAOS' if chaos else ''}"
          f"{' | HARDCORE' if hardcore else ''}"
          f"{' | INCIDENT' if incident else ''}")
    print(f"{'='*60}")

    model = MaskablePPO.load(model_path) if model_path else None

    rewards      = []
    lengths      = []
    timeouts     = []
    collisions   = []
    on_time      = []
    delays       = []
    ghat_waits   = []
    hp_delays    = []
    efficiencies = []
    latencies    = []

    for ep in range(num_episodes):
        env = TrainDispatchEnv()

        if stress:
            env.set_stress_mode(num_trains)
        else:
            env.set_difficulty(num_trains)
            
        if chaos or hardcore:
            env.set_chaos_mode(True, hardcore=hardcore)
            

        # Shield setup for RL+Shield mode
        optimizer = None
        if use_shield and SHIELD_AVAILABLE and model:
            env.reset(seed=seed + ep)
            try:
                shield = FeasibilityShield(
                    track_map=env.track_map,
                    station_nodes=env.station_nodes,
                    token_blocks=env.token_blocks,
                )
                env.attach_feasibility_shield(shield)
                optimizer = SmartOptimizer()
            except Exception:
                optimizer = None

        # Wrap for RL model
        if model:
            wrapped = ActionMasker(env, mask_fn)
            obs, _ = wrapped.reset(seed=seed + ep)
        else:
            obs, _ = env.reset(seed=seed + ep)

        if incident:
            env.apply_incident()

        ep_reward    = 0.0
        ep_len       = 0
        ep_collision = False
        done         = False
        ghat_wait_ep = 0

        ghat_nodes = _get_ghat_nodes(env)
        node_km = {
            nid: data.get('km', 0)
            for nid, data in env.track_map.items()
        }

        while not done:
            t0 = time.perf_counter()

            if model:
                action, _ = model.predict(obs, deterministic=True)
                if optimizer is not None:
                    action = optimizer.optimize_decision(
                        trains=env.trains,
                        ai_actions=action,
                        track_map=env.track_map,
                        ghat_token=env.ghat_token,
                        node_km=node_km,
                    )
            else:
                action = dispatch_fn(env, node_km)

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            if model:
                obs, rew, done, _, _ = wrapped.step(action)
            else:
                obs, rew, done, _, _ = env.step(action)

            ep_reward += rew
            ep_len    += 1

            if rew <= -70:
                ep_collision = True

            # Track trains idling near ghat (speed == 0, not dwelling)
            for t in env.trains:
                if (not t['finished']
                        and t['position'] in ghat_nodes
                        and t['speed'] == 0
                        and t.get('dwell_rem', 0) == 0
                        and t.get('banker_wait', 0) == 0):
                    ghat_wait_ep += 1

        # Per-train metrics
        metrics = _collect_episode_metrics(env, ep_len)

        rewards.append(ep_reward)
        lengths.append(ep_len)
        timeouts.append(int(ep_len >= 1490))
        collisions.append(int(ep_collision))
        on_time.append(metrics['on_time_pct'])
        delays.append(metrics['avg_delay'])
        ghat_waits.append(ghat_wait_ep)
        hp_delays.append(metrics['hp_delay'])
        efficiencies.append(metrics['efficiency'])

        if verbose and ep % 50 == 0:
            print(f"  Ep {ep:>3} | reward={ep_reward:>8.1f} | "
                  f"len={ep_len:>4} | on_time={metrics['on_time_pct']:.1f}%")

    summary = _make_summary(
        label, num_trains, num_episodes,
        rewards, lengths, timeouts, collisions,
        on_time, delays, ghat_waits, hp_delays, efficiencies,
        latencies if model else None,
    )

    print(f"\n{label} RESULTS:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch function adapters
# ─────────────────────────────────────────────────────────────────────────────

def _rule_dispatch(env, node_km):
    return rule_based_action(env.trains, env.track_map, node_km)


def _fcfs_dispatch(env, node_km):
    return fcfs_action(env.trains, env.track_map, env.schedule, node_km)


# ─────────────────────────────────────────────────────────────────────────────
# Table formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_table(results: list) -> str:
    """Format results as a markdown table with all metrics."""
    metrics = [
        ('avg_reward',      'Avg Reward'),
        ('std_reward',      'Std Reward'),
        ('min_reward',      'Min Reward (worst)'),
        ('p5_reward',       'P5 Reward (5th pctl)'),
        ('avg_ep_length',   'Avg Ep Length'),
        ('max_ep_length',   'Max Ep Length'),
        ('timeout_rate',    'Timeout Rate %'),
        ('collision_rate',  'Collision Rate %'),
        ('avg_on_time_pct', 'On Time %'),
        ('avg_delay',       'Avg Delay (steps)'),
        ('avg_ghat_wait',   'Ghat Wait (steps)'),
        ('avg_hp_delay',    'High-Prio Delay'),
        ('avg_efficiency',  'Efficiency (T/step)'),
    ]

    headers = ['Metric'] + [r['baseline'] for r in results]
    col_w = max(22, max(len(h) for h in headers) + 2)

    lines = []
    lines.append(' | '.join(h.ljust(col_w) for h in headers))
    lines.append('-|-'.join('-' * col_w for _ in headers))

    for metric_key, metric_label in metrics:
        row = [metric_label.ljust(col_w)]
        for r in results:
            val = r.get(metric_key, 'N/A')
            row.append(str(val).ljust(col_w))
        lines.append(' | '.join(row))

    # Improvement summary
    if len(results) >= 3:
        rule = results[0]
        agent = results[2]
        lines.append('')
        lines.append('─' * 60)
        lines.append('IMPROVEMENT SUMMARY (RL Agent vs Rule-Based):')
        lines.append('─' * 60)

        r_diff = agent.get('avg_reward', 0) - rule.get('avg_reward', 0)
        r_pct = (r_diff / abs(rule.get('avg_reward', 1))) * 100
        lines.append(f"  Reward:      {r_diff:+.2f} ({r_pct:+.1f}%)")

        ot_diff = agent.get('avg_on_time_pct', 0) - rule.get('avg_on_time_pct', 0)
        lines.append(f"  On-Time:     {ot_diff:+.1f} pp")

        std_diff = agent.get('std_reward', 0) - rule.get('std_reward', 0)
        lines.append(f"  Consistency: {std_diff:+.2f} std "
                      f"({'better' if std_diff < 0 else 'worse'})")

        wc_diff = agent.get('min_reward', 0) - rule.get('min_reward', 0)
        lines.append(f"  Worst-Case:  {wc_diff:+.2f} min reward "
                      f"({'better' if wc_diff > 0 else 'worse'})")

        hp_diff = agent.get('avg_hp_delay', 0) - rule.get('avg_hp_delay', 0)
        lines.append(f"  HP Delay:    {hp_diff:+.1f} steps "
                      f"({'better' if hp_diff < 0 else 'worse'})")

        rule_rewards = rule.get('raw_rewards', [])
        agent_rewards = agent.get('raw_rewards', [])
        if rule_rewards and agent_rewards:
            stat, p_val = ttest_ind(agent_rewards, rule_rewards, equal_var=False)
            lines.append(f"  T-Test (p):  {p_val:.4e} "
                          f"({'significant' if p_val < 0.05 else 'not significant'})")

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    model_path,
    num_trains=10,
    num_episodes=200,
    seed=42,
    save_dir='results',
    stress=False,
    chaos=False,
    hardcore=False,
    incident=False,
):
    os.makedirs(save_dir, exist_ok=True)

    mode_parts = []
    if stress: mode_parts.append("STRESS TEST")
    if chaos:  mode_parts.append("CHAOS")
    if hardcore: mode_parts.append("HARDCORE")
    if incident: mode_parts.append("INCIDENT")
    if not mode_parts: mode_parts.append("NORMAL")
    mode = " + ".join(mode_parts)
    print(f"\n{'#'*60}")
    print(f"# BENCHMARK TABLE — {num_trains} trains, {num_episodes} episodes")
    print(f"# Mode: {mode}  |  Seed: {seed}")
    print(f"{'#'*60}")

    all_results = []

    # 1. Rule-based
    rb = run_dispatcher(
        label='Rule-Based',
        num_trains=num_trains,
        num_episodes=num_episodes,
        seed=seed,
        stress=stress,
        chaos=chaos,
        hardcore=hardcore,
        incident=incident,
        dispatch_fn=_rule_dispatch,
    )
    all_results.append(rb)

    # 2. FCFS
    fcfs = run_dispatcher(
        label='FCFS',
        num_trains=num_trains,
        num_episodes=num_episodes,
        seed=seed,
        stress=stress,
        chaos=chaos,
        hardcore=hardcore,
        incident=incident,
        dispatch_fn=_fcfs_dispatch,
    )
    all_results.append(fcfs)

    # 3. RL Agent (no shield)
    agent = run_dispatcher(
        label='RL Agent',
        num_trains=num_trains,
        num_episodes=num_episodes,
        seed=seed,
        stress=stress,
        chaos=chaos,
        hardcore=hardcore,
        incident=incident,
        model_path=model_path,
        use_shield=False,
    )
    all_results.append(agent)

    # 4. RL Agent + Shield
    if SHIELD_AVAILABLE:
        agent_shield = run_dispatcher(
            label='RL + Shield',
            num_trains=num_trains,
            num_episodes=num_episodes,
            seed=seed,
            stress=stress,
            chaos=chaos,
            hardcore=hardcore,
            incident=incident,
            model_path=model_path,
            use_shield=True,
        )
        all_results.append(agent_shield)

    # Save full results
    with open(f'{save_dir}/benchmark_table.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print and save formatted table
    table = format_table(all_results)
    print(f"\n{'='*60}")
    print("BENCHMARK TABLE")
    print('=' * 60)
    print(table)

    with open(f'{save_dir}/benchmark_table.txt', 'w') as f:
        f.write(table)

    print(f"\nResults saved to {save_dir}/")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark all dispatchers with stress testing'
    )
    parser.add_argument('--model',    type=str, required=True)
    parser.add_argument('--trains',   type=int, default=10)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--stress',   action='store_true',
                        help='Use clustered-spawn stress test schedule')
    parser.add_argument('--chaos', action='store_true',
                        help='Enable chaos monkey mode')
    parser.add_argument('--hardcore', action='store_true',
                        help='Enable hardcore chaos monkey mode')
    parser.add_argument('--incident', action='store_true',
                        help='Enable random incident breakdowns')
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        num_trains=args.trains,
        num_episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir,
        stress=args.stress,
        chaos=args.chaos,
        hardcore=args.hardcore,
        incident=args.incident,
    )
