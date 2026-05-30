"""
smart_optimizer.py — Heuristic Safety Layer
Deterministic rule-engine running on top of RL step output.
Rewritten for CSMT-Manmad corridor with new track_map structure.

Key fixes from old version:
    - Node type detection uses track_map[node]['type'] not node_id >= 20000
    - Direction-aware main target resolution (UP trains move toward lower km)
    - Token block awareness (don't force-move into contested ghat)
    - Staging node handling (nodes 0 and 998)
    - No edge_id abstractions — uses node IDs throughout

Three layers (unchanged from design, implementation fixed):
    1. Anti-loitering  — force PROCEED when main line is clear
                         (FIXED: only fires when RL did NOT explicitly choose HOLD)
    2. Divert fallback — if DIVERT requested but loops full, use MAIN
    3. Collision gate  — FCFS capacity enforcement, highest priority first

Fix applied (2026-05-21):
    Layer 1 anti-loitering previously overrode the RL agent's deliberate HOLD
    decisions on mainline nodes. This destroyed learned anticipatory behaviour
    (e.g. holding a goods train to let a Rajdhani pass). The fix adds a check:
    anti-loitering only fires when the RL agent's ORIGINAL action was not 0
    (i.e. the env or dwell logic forced the hold, not the agent itself).
"""

import logging
from collections import defaultdict
from typing import Optional
import numpy as np

_log = logging.getLogger("SmartOptimizer")

# Node types considered "main line" — trains should not idle here
MAINLINE_TYPES = {'MAIN_BLOCK', 'GHAT_BLOCK', 'SWITCH', 'ORIGIN'}

# Node types considered "holding areas" — trains may idle here
HOLDING_TYPES = {'PLATFORM', 'LOOP', 'DESTINATION'}


class SmartOptimizer:
    """
    Heuristic interlocking system — deterministic, no solver needed.
    Runs in O(N log N) time. Always completes in <1ms.

    Usage (inference):
        optimizer = SmartOptimizer()
        safe_actions = optimizer.optimize_decision(
            trains=env.trains,
            ai_actions=raw_actions,
            track_map=env.track_map,
            ghat_token=env.ghat_token,
        )
    """

    def __init__(self):
        _log.info("SmartOptimizer initialized")

    def optimize_decision(
        self,
        trains: list,
        ai_actions: np.ndarray,
        track_map: dict,
        ghat_token=None,           # GhatTokenSystem | None
        node_km: dict = None,      # node_id → km, for direction resolution
    ) -> np.ndarray:
        """
        Apply heuristic safety rules on top of RL action proposals.

        Parameters
        ----------
        trains      : env.trains list
        ai_actions  : raw actions from RL policy, shape (MAX_TRAINS_CAPACITY,)
        track_map   : env.track_map dict
        ghat_token  : env.ghat_token GhatTokenSystem (optional)
        node_km     : node→km dict for direction-aware target resolution

        Returns
        -------
        np.ndarray — safe actions, same shape as ai_actions
        """
        safe_actions = ai_actions.copy().astype(int)
        node_km      = node_km or {}

        # ── Snapshot current occupancy ────────────────────────────────────
        current_occ = defaultdict(int)
        for t in trains:
            p = t['position']
            if p not in (0, 998, 999):
                current_occ[p] += 1

        # ── FCFS claim tracker — updated as trains are processed ──────────
        # Highest priority trains processed first, they claim capacity first
        claimed_occ = defaultdict(int)

        sorted_indices = sorted(
            range(len(trains)),
            key=lambda k: trains[k].get('priority', 0),
            reverse=True,
        )

        for i in sorted_indices:
            train = trains[i]
            pos   = train['position']
            act   = int(safe_actions[i])

            # The RL agent's original action before any override.
            # Used by anti-loitering to detect deliberate HOLD decisions.
            rl_original_action = int(ai_actions[i]) if i < len(ai_actions) else 0

            # Skip finished trains
            if train['finished'] or pos == 999:
                claimed_occ[pos] += 1
                continue

            # Skip staging nodes
            if pos in (0, 998):
                continue

            node_data  = track_map.get(pos, {})
            node_type  = node_data.get('type', 'MAIN_BLOCK')
            next_opts  = node_data.get('next', [])
            direction  = train.get('direction', 'DOWN')
            is_token   = node_data.get('token_block', False)

            if not next_opts:
                claimed_occ[pos] += 1
                continue

            # Direction-aware main target
            main_target = self._resolve_main_target(
                pos, next_opts, direction, node_km, track_map
            )
            loop_targets = [n for n in next_opts if n != main_target]

            # ── Layer 1: Anti-loitering ───────────────────────────────────
            # Only override HOLD → PROCEED when the RL agent did NOT
            # explicitly choose HOLD. If rl_original_action == 0 the agent
            # made a deliberate decision to wait (strategic hold) and we
            # must respect it. We only fire anti-loitering when something
            # else (dwell logic, env guard) forced the current act to 0
            # while the agent actually wanted to move.
            if act == 0 and node_type in MAINLINE_TYPES and rl_original_action != 0:
                main_occ = current_occ[main_target] + claimed_occ[main_target]
                main_cap = track_map.get(main_target, {}).get('capacity', 1)

                # Don't force into token block if opposing direction holds it
                token_blocked = False
                if main_target in self._get_token_set(ghat_token):
                    if ghat_token and not ghat_token.can_enter(train['id'], direction):
                        token_blocked = True

                if main_occ < main_cap and not token_blocked:
                    act = 1
                    safe_actions[i] = 1
                    _log.debug(
                        f"Anti-loiter: {train['id']} forced PROCEED from "
                        f"{node_type} (rl_original={rl_original_action})"
                    )

            # ── Layer 1b: Loop wake-up ────────────────────────────────────
            # Loop wake-up is intentional even when RL chose HOLD: a train
            # sitting in a loop after dwell is done should always exit when
            # the main line is free — there is no strategic reason to stay.
            elif act == 0 and node_type in HOLDING_TYPES:
                main_occ = current_occ[main_target] + claimed_occ[main_target]
                main_cap = track_map.get(main_target, {}).get('capacity', 1)
                # Only wake up if dwell is done (dwell_rem == 0)
                if (main_occ < main_cap
                        and train.get('dwell_rem', 0) == 0
                        and claimed_occ[main_target] == 0):
                    act = 1
                    safe_actions[i] = 1
                    _log.debug(f"Loop wake-up: {train['id']} exiting {node_type}")

            # ── Layer 2: Divert fallback ──────────────────────────────────
            elif act == 2:
                loop_available = False
                for ln in loop_targets:
                    ln_occ = current_occ[ln] + claimed_occ[ln]
                    ln_cap = track_map.get(ln, {}).get('capacity', 1)
                    if ln_occ < ln_cap:
                        loop_available = True
                        break

                if not loop_available:
                    # All loops full — fall back to MAIN
                    act = 1
                    safe_actions[i] = 1
                    _log.debug(f"Divert fallback: {train['id']} loops full, using MAIN")

            # ── Layer 3: Collision gate ───────────────────────────────────
            # Predict target position
            if act == 0:
                target_pos = pos   # staying put
            elif act == 1:
                target_pos = main_target
            else:  # act == 2
                target_pos = pos   # default if no loop found
                for ln in loop_targets:
                    ln_occ = current_occ[ln] + claimed_occ[ln]
                    ln_cap = track_map.get(ln, {}).get('capacity', 1)
                    if ln_occ < ln_cap:
                        target_pos = ln
                        break

            # Capacity enforcement
            if target_pos not in (0, 998, 999) and target_pos != pos:
                target_cap = track_map.get(target_pos, {}).get('capacity', 1)
                total_occ  = current_occ[target_pos] + claimed_occ[target_pos]

                if total_occ >= target_cap:
                    # Block is full — force HOLD
                    safe_actions[i] = 0
                    target_pos      = pos
                    _log.debug(
                        f"Collision gate: {train['id']} blocked from "
                        f"node {target_pos} (occ={total_occ}/{target_cap})"
                    )

            # Update claim tracker
            claimed_occ[target_pos] += 1

        return safe_actions

    def _resolve_main_target(
        self,
        pos: int,
        next_opts: list,
        direction: str,
        node_km: dict,
        track_map: dict,
    ) -> int:
        """Direction-aware main target resolution."""
        if direction == 'UP':
            my_km = node_km.get(pos, 0)
            candidates = [
                n for n in next_opts
                if node_km.get(n, 0) <= my_km
            ]
            if candidates:
                return min(candidates, key=lambda n: node_km.get(n, 0))
        return next_opts[0]

    def _get_token_set(self, ghat_token) -> set:
        """Return token block node IDs or empty set if no token system."""
        if ghat_token is None:
            return set()
        return ghat_token.token_block_ids
