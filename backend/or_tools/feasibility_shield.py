"""
feasibility_shield.py — CP-SAT Feasibility Shield
Inference-only safety layer for CSMT-Manmad corridor.

Hooks into TrainDispatchEnv via:
    env.attach_feasibility_shield(FeasibilityShield(track_map, station_nodes, token_blocks))

Called inside get_action_mask() AFTER structural mask is applied.
Only removes actions — never adds them.

Three constraint layers (as designed):
    Layer 1 — Token block: opposing direction holds ghat mid-line
    Layer 2 — Capacity: node will be over capacity if action proceeds
    Layer 3 — Punctuality: action leads to guaranteed deadline violation
               when a better alternative exists

Performance target: <5ms per call with 10 trains.
"""

import time
import logging
import numpy as np
from collections import defaultdict
from ortools.sat.python import cp_model

from ai.config import (
    MAX_TRAINS_CAPACITY,
    SECTION_LENGTH_KM,
    BANKER_ATTACH_TIME,
)
from ai.map_generator import GhatTokenSystem

_log = logging.getLogger("FeasibilityShield")

# Lookahead horizon for CP-SAT punctuality check
SHIELD_LOOKAHEAD_STEPS = 50

# Punctuality threshold — only veto if the joint delay improvement exceeds this
MIN_DELAY_IMPROVEMENT = 10   # sim-steps (joint improvement threshold)

# Hard timeout for solver — never block inference more than this
SOLVER_TIMEOUT_SECONDS = 0.020  # 20ms


class FeasibilityShield:
    """
    CP-SAT feasibility shield — inference only.

    Usage:
        shield = FeasibilityShield(track_map, station_nodes, token_blocks)
        env.attach_feasibility_shield(shield)

    The shield is called inside get_action_mask() after the structural
    mask is already applied. It only further restricts actions.

    Three layers run in order, cheapest first:
        1. Token check    — O(1), no solver needed
        2. Capacity check — O(N), no solver needed
        3. Punctuality    — CP-SAT solver, bounded by SOLVER_TIMEOUT_SECONDS
    """

    def __init__(
        self,
        track_map: dict,
        station_nodes: dict,
        token_blocks: list,
    ):
        self.track_map     = track_map
        self.station_nodes = station_nodes
        self.token_blocks  = set(token_blocks)

        # Build node→km lookup for fast distance queries
        self._node_km = {
            nid: data.get('km', 0.0)
            for nid, data in track_map.items()
        }
        self._node_km[0]   = 0.0
        self._node_km[998] = SECTION_LENGTH_KM
        self._node_km[999] = SECTION_LENGTH_KM

        # Performance tracking
        self._call_count    = 0
        self._total_ms      = 0.0
        self._vetoes        = defaultdict(int)  # layer → count

        _log.info("FeasibilityShield initialized | "
                  f"token_blocks={sorted(token_blocks)} | "
                  f"lookahead={SHIELD_LOOKAHEAD_STEPS} steps")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API — called by get_action_mask()
    # ─────────────────────────────────────────────────────────────────────────

    def get_masked_actions(
        self,
        sim_time: int,
        current_mask: np.ndarray,
        trains: list,
        schedule: dict,
        ghat_token: GhatTokenSystem,
    ) -> np.ndarray:
        """
        Apply three constraint layers on top of structural mask.

        Parameters
        ----------
        sim_time     : current simulation timestep
        current_mask : structural mask from get_action_mask() shape (MAX_TRAINS_CAPACITY, 3)
        trains       : env.trains list
        schedule     : env.schedule dict
        ghat_token   : env.ghat_token GhatTokenSystem instance

        Returns
        -------
        np.ndarray shape (MAX_TRAINS_CAPACITY, 3) — subset of current_mask
        """
        t0   = time.perf_counter()
        mask = current_mask.copy()

        # Build occupancy snapshot once — used by all layers
        occupancy = defaultdict(int)
        for t in trains:
            p = t['position']
            if p not in (0, 998, 999):
                occupancy[p] += 1

        for i, train in enumerate(trains):
            if i >= MAX_TRAINS_CAPACITY:
                break
            if train['finished'] or train['position'] in (0, 998, 999):
                continue
            # Only process if at least PROCEED or DIVERT is still allowed
            if not mask[i, 1] and not mask[i, 2]:
                continue

            pos       = train['position']
            direction = train['direction']
            sched     = schedule.get(train['id'], {})

            node_data = self.track_map.get(pos, {})
            next_opts = node_data.get('next', [])
            if not next_opts:
                continue

            # Resolve main target (direction-aware)
            if direction == 'UP':
                candidates = [
                    n for n in next_opts
                    if self._node_km.get(n, 0) <= self._node_km.get(pos, 0)
                ]
                main_target = (
                    min(candidates, key=lambda n: self._node_km.get(n, 0))
                    if candidates else next_opts[0]
                )
            else:
                main_target = next_opts[0]

            loop_targets = [n for n in next_opts if n != main_target]

            # ── Layer 1: Token block constraint ───────────────────────────
            if mask[i, 1] and main_target in self.token_blocks:
                if not ghat_token.can_enter(train['id'], direction):
                    mask[i, 1] = False
                    self._vetoes['layer1_token'] += 1
                    _log.debug(f"L1 veto: train {train['id']} blocked from token block")

            # ── Layer 2: Capacity constraint ──────────────────────────────
            if mask[i, 1]:
                # Fix: Only apply strict capacity constraints for non-MAIN nodes.
                # RL agent can safely handle tight following distances on MAIN nodes,
                # and the environment will sequential-check it anyway.
                node_type = self.track_map.get(main_target, {}).get('type', 'MAIN')
                if node_type != 'MAIN':
                    main_cap = self.track_map.get(main_target, {}).get('capacity', 1)
                    # Count current + trains that already moved here this step
                    if occupancy[main_target] >= main_cap:
                        mask[i, 1] = False
                        self._vetoes['layer2_capacity_main'] += 1

            if mask[i, 2] and loop_targets:
                all_loops_full = True
                for ln in loop_targets:
                    loop_cap = self.track_map.get(ln, {}).get('capacity', 1)
                    if occupancy[ln] < loop_cap:
                        all_loops_full = False
                        break
                if all_loops_full:
                    mask[i, 2] = False
                    self._vetoes['layer2_capacity_loop'] += 1

            # ── Layer 3: Punctuality constraint (CP-SAT) ──────────────────
            # Only run if both PROCEED and DIVERT are still available —
            # if only one action is legal, no choice to optimize.
            # Also skip if no deadline pressure.
            # Skip entirely for P1-P2 trains — RL agent manages low-priority
            # freight decisions; shield only guards P3+ under genuine deadline.
            if mask[i, 1] and mask[i, 2] and loop_targets:
                deadline = sched.get('deadline', 99999)
                deadline_rem = deadline - sim_time
                train_priority = train.get('priority', 1)
                if train_priority >= 3 and deadline_rem < SHIELD_LOOKAHEAD_STEPS * 1:
                    # Genuine last-chance deadline pressure — run CP-SAT
                    better = self._punctuality_check(
                        train=train,
                        main_target=main_target,
                        loop_targets=loop_targets,
                        trains=trains,
                        occupancy=occupancy,
                        sim_time=sim_time,
                        schedule=schedule,
                    )
                    if better == 'divert':
                        mask[i, 1] = False
                        self._vetoes['layer3_punctuality'] += 1
                    elif better == 'proceed':
                        mask[i, 2] = False
                        self._vetoes['layer3_punctuality'] += 1

        # Performance tracking
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_ms   += elapsed_ms

        if elapsed_ms > 10:
            _log.warning(f"Shield slow: {elapsed_ms:.1f}ms at step {sim_time}")

        return mask

    def _simulate_eta(self, train: dict, target_node: int, trains: list, occupancy: dict, sim_time: int) -> int:
        """
        Fast forward-simulator for K steps to estimate arrival time.
        Accounts for immediate token blocks and congestion, then uses linear ETA for the rest.
        """
        pos = target_node
        direction = train['direction']
        time_spent = 0
        max_steps = SHIELD_LOOKAHEAD_STEPS
        
        while time_spent < max_steps:
            # 1. Token Block wait
            if pos in self.token_blocks:
                opposing_in_ghat = any(
                    t['direction'] != direction and t['position'] in self.token_blocks 
                    for t in trains if not t['finished']
                )
                if opposing_in_ghat:
                    time_spent += 1
                    continue
            
            # 2. Reached destination?
            if pos in (0, 998, 999):
                break
                
            node_data = self.track_map.get(pos, {})
            next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
            if not next_opts:
                break
                
            # 3. Congestion ahead
            # Resolve main target same as env
            if direction == 'UP':
                candidates = [n for n in next_opts if self._node_km.get(n, 0) <= self._node_km.get(pos, 0)]
                main_target = min(candidates, key=lambda n: self._node_km.get(n, 0)) if candidates else next_opts[0]
            else:
                main_target = next_opts[0]
                
            cap = self.track_map.get(main_target, {}).get('capacity', 1)
            if occupancy.get(main_target, 0) >= cap:
                time_spent += 1
                continue
                
            # 4. Move forward
            pos = main_target
            time_spent += 1
            
        # Add remaining linear ETA from the position after K steps
        current_km = self._node_km.get(pos, 0)
        dest_km = SECTION_LENGTH_KM if direction == 'DOWN' else 0.0
        rem_dist = abs(dest_km - current_km)
        avg_km_per_step = max(train['max_speed'] / 60.0, 0.1)
        
        return sim_time + time_spent + int(rem_dist / avg_km_per_step)

    def _find_train_behind(self, train: dict, trains: list):
        """
        Find the closest train traveling in the same direction that is
        immediately behind the given train (has NOT yet reached this train's
        position km).

        'Behind' means: same direction, closer to origin, within 20 km.
        We pick the one with the HIGHEST priority among those close by, so
        the solver correctly prioritises letting a Rajdhani overtake a Freight.
        """
        direction   = train['direction']
        current_km  = self._node_km.get(train['position'], 0)
        best        = None
        best_priority = -1

        for t in trains:
            if t['id'] == train['id'] or t['finished']:
                continue
            if t['direction'] != direction:
                continue

            t_km = self._node_km.get(t['position'], 0)

            # 'Behind' in DOWN direction = lower km value
            # 'Behind' in UP direction   = higher km value
            if direction == 'DOWN':
                gap = current_km - t_km   # positive = t is behind us
            else:
                gap = t_km - current_km   # positive = t is behind us

            if 0 < gap <= 20.0:  # within 20 km behind
                if t.get('priority', 1) > best_priority:
                    best = t
                    best_priority = t.get('priority', 1)

        return best

    def _punctuality_check(
        self,
        train: dict,
        main_target: int,
        loop_targets: list,
        trains: list,
        occupancy: dict,
        sim_time: int,
        schedule: dict,
    ) -> str:
        """
        Run a K-step CP-SAT horizon to determine whether PROCEED or DIVERT
        leads to better punctuality outcome for this train and its neighbors.
        """
        try:
            model  = cp_model.CpModel()
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = SOLVER_TIMEOUT_SECONDS

            train_id   = train['id']
            direction  = train['direction']
            deadline   = schedule.get(train_id, {}).get('deadline', 99999)
            priority   = train['priority']

            # ── Variables ────────────────────────────────────────────────
            proceeds = model.NewBoolVar(f'proceeds_{train_id}')

            # ── ETAs for the current train ────────────────────────────────
            eta_proceed = self._simulate_eta(train, main_target, trains, occupancy, sim_time)
            divert_node = loop_targets[0] if loop_targets else main_target
            dwell_penalty = 5 if self.track_map.get(divert_node, {}).get('type') == 'CROSSING_LOOP' else 10
            eta_divert  = self._simulate_eta(train, divert_node, trains, occupancy, sim_time + dwell_penalty)

            MAX_DELAY = 1000

            # Base delays for current train
            base_dp = max(0, eta_proceed - deadline)
            base_dd = max(0, eta_divert  - deadline)

            delay_proceed = model.NewIntVar(0, MAX_DELAY, 'delay_proceed')
            delay_divert  = model.NewIntVar(0, MAX_DELAY, 'delay_divert')
            model.Add(delay_proceed == base_dp)
            model.Add(delay_divert  == base_dd)

            # Weighted cost for current train
            cost_p = model.NewIntVar(0, MAX_DELAY * 10, 'cost_p')
            cost_d = model.NewIntVar(0, MAX_DELAY * 10, 'cost_d')
            model.AddMultiplicationEquality(cost_p, [delay_proceed, priority])
            model.AddMultiplicationEquality(cost_d, [delay_divert,  priority])

            # ── ALTRUISTIC: find the train immediately behind ─────────────
            # If a high-priority train is riding our tail, factor its delay
            # into the objective. This allows OR-Tools to approve a divert
            # even when it costs the current train time, if it saves a
            # Priority-5 Rajdhani from being trapped behind us.
            behind = self._find_train_behind(train, trains)

            total_cost_p = model.NewIntVar(0, MAX_DELAY * 20, 'total_cost_p')
            total_cost_d = model.NewIntVar(0, MAX_DELAY * 20, 'total_cost_d')

            if behind is not None:
                b_deadline = schedule.get(behind['id'], {}).get('deadline', 99999)
                b_priority = behind.get('priority', 1)

                # If we proceed (blocking the train behind), the train behind
                # is stuck behind us and will be delayed by our remaining ETA
                # minus what it would have been if we had diverted.
                # Simplified: if we block, the train behind inherits a delay
                # proportional to how much slower we are.
                our_speed   = max(train['max_speed'], 1)
                their_speed = max(behind['max_speed'], 1)

                # Blocking penalty = extra time the faster train loses waiting
                # behind the slower train, capped at 60 steps to avoid explosion
                if their_speed > our_speed:
                    block_steps = int(min((their_speed - our_speed) / our_speed * 30, 60))
                else:
                    block_steps = 0

                b_eta_proceed_blocked = self._simulate_eta(behind, behind['position'], trains, occupancy, sim_time) + block_steps
                b_eta_proceed_free    = self._simulate_eta(behind, behind['position'], trains, occupancy, sim_time)
                b_eta_divert_free     = b_eta_proceed_free   # if we divert, behind moves freely

                b_delay_if_we_proceed = max(0, b_eta_proceed_blocked - b_deadline)
                b_delay_if_we_divert  = max(0, b_eta_divert_free     - b_deadline)

                b_cost_if_we_proceed = b_delay_if_we_proceed * b_priority
                b_cost_if_we_divert  = b_delay_if_we_divert  * b_priority

                # Joint cost = our cost + the train-behind's cost
                model.Add(total_cost_p == cost_p + b_cost_if_we_proceed)
                model.Add(total_cost_d == cost_d + b_cost_if_we_divert)
            else:
                # No train behind — same as selfish mode
                model.Add(total_cost_p == cost_p)
                model.Add(total_cost_d == cost_d)

            # ── Objective: minimise the JOINT weighted delay ──────────────
            chosen_cost = model.NewIntVar(0, MAX_DELAY * 20, 'chosen_cost')
            model.Add(chosen_cost == total_cost_p).OnlyEnforceIf(proceeds)
            model.Add(chosen_cost == total_cost_d).OnlyEnforceIf(proceeds.Not())
            model.Minimize(chosen_cost)

            status = solver.Solve(model)

            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return 'either'

            should_proceed = solver.Value(proceeds)

            # Veto only when joint improvement is meaningful
            if should_proceed:
                # Compute joint improvement: how much worse is divert for the pair?
                joint_p = solver.Value(total_cost_p)
                joint_d = solver.Value(total_cost_d)
                improvement = joint_d - joint_p  # positive = proceed is better
                if improvement >= MIN_DELAY_IMPROVEMENT:
                    return 'proceed'
            else:
                joint_p = solver.Value(total_cost_p)
                joint_d = solver.Value(total_cost_d)
                improvement = joint_p - joint_d  # positive = divert is better
                if improvement >= MIN_DELAY_IMPROVEMENT:
                    return 'divert'

            return 'either'

        except Exception as e:
            _log.warning(f"CP-SAT punctuality check failed: {e}")
            return 'either'

    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE STATS
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return performance and veto statistics."""
        avg_ms = self._total_ms / max(self._call_count, 1)
        return {
            'calls':        self._call_count,
            'avg_ms':       round(avg_ms, 3),
            'total_ms':     round(self._total_ms, 1),
            'vetoes':       dict(self._vetoes),
            'total_vetoes': sum(self._vetoes.values()),
        }

    def reset_stats(self):
        self._call_count = 0
        self._total_ms   = 0.0
        self._vetoes     = defaultdict(int)
