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
SHIELD_LOOKAHEAD_STEPS = 5

# Punctuality threshold — only veto if delay improvement exceeds this
MIN_DELAY_IMPROVEMENT = 10   # sim-steps

# Hard timeout for solver — never block inference more than this
SOLVER_TIMEOUT_SECONDS = 0.004  # 4ms


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
            if mask[i, 1] and mask[i, 2] and loop_targets:
                deadline = sched.get('deadline', 99999)
                deadline_rem = deadline - sim_time
                if deadline_rem < SHIELD_LOOKAHEAD_STEPS * 3:
                    # Deadline pressure — run CP-SAT to pick better action
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

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 3 — CP-SAT punctuality optimizer
    # ─────────────────────────────────────────────────────────────────────────

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

        Returns
        -------
        'proceed'  — DIVERT should be masked (proceed is better)
        'divert'   — PROCEED should be masked (divert is better)
        'either'   — no meaningful difference, keep both
        """
        try:
            model  = cp_model.CpModel()
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = SOLVER_TIMEOUT_SECONDS

            K          = SHIELD_LOOKAHEAD_STEPS
            train_id   = train['id']
            direction  = train['direction']
            deadline   = schedule.get(train_id, {}).get('deadline', 99999)
            priority   = train['priority']

            # ── Variables ────────────────────────────────────────────────
            # Binary: does this train PROCEED (1) or DIVERT (0)?
            proceeds = model.NewBoolVar(f'proceeds_{train_id}')

            # Estimated position km after K steps
            # If PROCEED: moves toward destination at ~1 block/step
            proceed_km = self._node_km.get(main_target, 0)
            divert_km  = self._node_km.get(
                loop_targets[0] if loop_targets else main_target, 0
            )

            # Km remaining to destination
            dest_km = SECTION_LENGTH_KM if direction == 'DOWN' else 0.0

            # Estimated arrival step if proceeding vs diverting
            # Simple linear estimate: remaining_km / avg_speed_per_step
            avg_km_per_step = max(train['max_speed'] / 60.0, 0.1)

            if direction == 'DOWN':
                rem_proceed = max(dest_km - proceed_km, 0)
                rem_divert  = max(dest_km - divert_km, 0)
            else:
                rem_proceed = max(proceed_km - dest_km, 0)
                rem_divert  = max(divert_km  - dest_km, 0)

            steps_proceed = int(rem_proceed / avg_km_per_step)
            steps_divert  = int(rem_divert  / avg_km_per_step)

            eta_proceed = sim_time + steps_proceed
            eta_divert  = sim_time + steps_divert

            # Delay variables (bounded integers)
            MAX_DELAY = 500
            delay_proceed = model.NewIntVar(0, MAX_DELAY, 'delay_proceed')
            delay_divert  = model.NewIntVar(0, MAX_DELAY, 'delay_divert')

            model.Add(delay_proceed == max(0, eta_proceed - deadline))
            model.Add(delay_divert  == max(0, eta_divert  - deadline))

            # Weighted delay cost
            cost_proceed = delay_proceed * priority
            cost_divert  = delay_divert  * priority

            # ── Constraints ───────────────────────────────────────────────
            # Check if main target is congested by same-direction trains
            congestion_ahead = sum(
                1 for t in trains
                if not t['finished']
                and t['direction'] == direction
                and t['position'] not in (0, 998, 999)
                and abs(
                    self._node_km.get(t['position'], 0) -
                    self._node_km.get(main_target, 0)
                ) < 5.0   # within 5km
                and t['id'] != train_id
            )

            # If high congestion ahead, divert has lower effective cost
            if congestion_ahead >= 2:
                # Add congestion penalty to proceed cost
                model.Add(delay_proceed >= congestion_ahead * 3)

            # ── Objective ─────────────────────────────────────────────────
            # Minimize: proceeds * cost_proceed + (1-proceeds) * cost_divert
            # Simplified: minimize the chosen action's delay cost
            chosen_delay = model.NewIntVar(0, MAX_DELAY * 6, 'chosen_delay')
            model.Add(chosen_delay == cost_proceed).OnlyEnforceIf(proceeds)
            model.Add(chosen_delay == cost_divert).OnlyEnforceIf(proceeds.Not())
            model.Minimize(chosen_delay)

            status = solver.Solve(model)

            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return 'either'

            should_proceed = solver.Value(proceeds)

            # Only veto if the improvement is meaningful
            if should_proceed:
                improvement = max(0, eta_divert - eta_proceed)
                if improvement >= MIN_DELAY_IMPROVEMENT:
                    return 'proceed'
            else:
                improvement = max(0, eta_proceed - eta_divert)
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
