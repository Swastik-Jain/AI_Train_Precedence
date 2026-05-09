"""
train_env.py — TrainDispatchEnv
CSMT → Manmad corridor, Central Railway Bhusawal-Kalyan division.

Key changes from toy map version:
  - Bidirectional traffic: UP (Manmad→CSMT) and DOWN (CSMT→Manmad)
  - GhatTokenSystem: Kasara↔Igatpuri mid-line token block
  - km-based observation space (23 features) — no MAX_LOOKAHEAD
  - Schedule-driven stops: trains only dwell at their scheduled stations
  - Banker loco logic: attach/detach at Kasara and Igatpuri
  - O(1) occupancy via Counter updated incrementally
  - Correct spawn points: DOWN trains spawn at CSMT end, UP at Manmad end
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import logging
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.config import (
    ACTIVE_FLEET, SCHEDULE,
    MAX_TRAINS_CAPACITY, MAX_SPEED, SECTION_LENGTH_KM,
    DWELL_TIME_PLATFORM, DWELL_TIME_LOOP,
    BANKER_ATTACH_TIME, BANKER_DETACH_TIME,
    DANGER_HORIZON_KM, SPACING_HORIZON_KM,
    generate_daily_schedule,
)
from ai.map_generator import (
    generate_realistic_section,
    GhatTokenSystem,
    SECTION_LENGTH_KM as MAP_SECTION_KM,
)

_log = logging.getLogger("TrainDispatchEnv")

# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION SPACE — 23 features per train
# ─────────────────────────────────────────────────────────────────────────────
#
# Category 1 — Self (6 features)
#   0  my_speed            / MAX_SPEED
#   1  my_priority         / 6.0
#   2  direction           0=DOWN, 1=UP
#   3  route_progress      km_done / SECTION_LENGTH_KM
#   4  delay_norm          min(current_delay, 60) / 60.0
#   5  dwell_norm          dwell_rem / DWELL_TIME_PLATFORM
#
# Category 2 — Forward danger (4 features)
#   6  signal_value        0=clear, 0.5=caution, 1.0=danger
#   7  dist_to_danger_norm / DANGER_HORIZON_KM
#   8  dist_to_lead_norm   / SPACING_HORIZON_KM
#   9  lead_speed_norm     / MAX_SPEED
#
# Category 3 — Opposing traffic (5 features)
#   10 opposing_present    0/1
#   11 opposing_dist_norm  / SECTION_LENGTH_KM
#   12 opposing_speed_norm / MAX_SPEED
#   13 opposing_prio_norm  / 6.0
#   14 token_status        0=free, 0.5=held_same_dir, 1.0=held_opposing
#
# Category 4 — Station awareness (4 features)
#   15 dist_to_next_stop_norm  / SECTION_LENGTH_KM
#   16 at_banker_point      0/1  (Kasara or Igatpuri)
#   17 banker_wait_norm     banker_wait / BANKER_ATTACH_TIME
#   18 platform_avail       free_platforms / total_platforms at next stop
#
# Category 5 — Section load (3 features)
#   19 section_load         active_trains / MAX_TRAINS_CAPACITY
#   20 trains_same_dir_norm count_same_direction / MAX_TRAINS_CAPACITY
#   21 deadline_norm        min(deadline_rem, 500) / 500.0
#
# Category 6 — Schedule (1 feature)
#   22 lead_priority_norm  priority of highest-prio train ahead / 6.0
#
N_FEATURES = 23

# Ghost train padding — represents an empty slot
_GHOST_OBS = np.zeros(N_FEATURES, dtype=np.float32)
_GHOST_OBS[7]  = 1.0   # dist_to_danger: far
_GHOST_OBS[8]  = 1.0   # dist_to_lead: far
_GHOST_OBS[11] = 1.0   # opposing_dist: far


class TrainDispatchEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(MAX_TRAINS_CAPACITY, N_FEATURES),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([3] * MAX_TRAINS_CAPACITY)

        # State — populated in reset()
        self.trains             = []
        self.active_fleet       = [t.copy() for t in ACTIVE_FLEET]
        self.schedule           = {k: v.copy() for k, v in SCHEDULE.items()}
        self.track_map          = {}
        self.loop_sections      = []
        self.end_node           = 999
        self.station_nodes      = {}
        self.token_blocks       = []
        self.ghat_token         = None
        self.sim_time           = 0

        # O(1) occupancy counter — updated incrementally in step()
        self._occupancy         = Counter()

        # OR-solver shield (optional)
        self._feasibility_shield = None

        # Chaos monkey (disabled by default)
        self.chaos_mode         = False
        self.chaos_delay_prob   = 0.30
        self.chaos_delay_min    = 1
        self.chaos_delay_max    = 10
        self.chaos_speed_snag   = True
        self.chaos_speed_factor = 0.80

        self.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def set_difficulty(self, num_trains: int):
        fleet, sched = generate_daily_schedule(num_trains)
        self.active_fleet = fleet
        self.schedule     = sched
        self.reset()
        print(f"🔹 Difficulty set: {num_trains} trains")

    def attach_feasibility_shield(self, shield):
        self._feasibility_shield = shield
        status = "attached" if shield is not None else "detached"
        print(f"🛡️  FeasibilityShield {status}.")

    def set_chaos_mode(self, enabled=True, delay_prob=0.30,
                       delay_min=1, delay_max=10,
                       speed_snag=True, speed_factor=0.80):
        self.chaos_mode         = enabled
        self.chaos_delay_prob   = delay_prob
        self.chaos_delay_min    = delay_min
        self.chaos_delay_max    = delay_max
        self.chaos_speed_snag   = speed_snag
        self.chaos_speed_factor = speed_factor
        state = "ENABLED 🐒" if enabled else "DISABLED ✅"
        print(f"⚡ Chaos Monkey {state} | delay_prob={delay_prob:.0%} "
              f"delay={delay_min}–{delay_max}m speed_snag={speed_snag}({speed_factor:.0%})")

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim_time = 0

        # Fresh fleet + schedule each episode (randomized)
        self.active_fleet, self.schedule = generate_daily_schedule(
            len(self.active_fleet)
        )

        # Deep-copy fleet to working trains list
        self.trains = []
        for t in self.active_fleet:
            train = t.copy()
            train['speed']           = 0
            train['target_speed']    = 0
            train['delay']           = 0
            train['idle_time']       = 0
            train['dwell_rem']       = 0
            train['finished']        = False
            train['banker_attached'] = False
            train['banker_wait']     = 0
            train['visited_nodes']   = set()
            # Spawn position depends on direction
            # DOWN (CSMT→Manmad): spawn at node 0 (CSMT end)
            # UP   (Manmad→CSMT): spawn at node 998 (Manmad end staging)
            train['position'] = 0 if train['direction'] == 'DOWN' else 998
            self.trains.append(train)

        # Fresh map each episode
        (self.track_map,
         self.loop_sections,
         self.end_node,
         self.station_nodes,
         self.token_blocks) = generate_realistic_section()

        # Add UP-direction staging node (mirror of node 0 for DOWN trains)
        # Node 998: UP trains wait here before entering Manmad switch_in
        manmad_switch_in = self.station_nodes['MANMAD']['switch_in']
        self.track_map[998] = {
            'type':        'ORIGIN',
            'speed':       0,
            'capacity':    99,
            'next':        [manmad_switch_in],
            'km':          SECTION_LENGTH_KM,
            'station':     None,
            'token_block': False,
            'gradient':    False,
        }

        # Token system for Kasara-Igatpuri ghat
        self.ghat_token = GhatTokenSystem(self.token_blocks)

        # O(1) occupancy — rebuild from current positions
        self._occupancy = Counter(
            t['position'] for t in self.trains
            if t['position'] not in [0, 998, 999]
        )

        # Physics arrays
        self._train_speeds = np.zeros(MAX_TRAINS_CAPACITY, dtype=np.float32)
        self._movement_acc = np.zeros(MAX_TRAINS_CAPACITY, dtype=np.float32)

        if self.chaos_mode:
            self._apply_chaos()

        return self._get_observation(), {}

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def get_node_km(self, node_id: int) -> float:
        """Return physical km position of a node. O(1) from track_map."""
        if node_id in (0, 998):
            # Staging nodes: DOWN starts at 0km, UP starts at section end
            return 0.0 if node_id == 0 else SECTION_LENGTH_KM
        if node_id == 999:
            return SECTION_LENGTH_KM
        return self.track_map.get(node_id, {}).get('km', 0.0)

    def get_node_occupancy(self, node_id: int) -> int:
        """O(1) occupancy lookup."""
        return self._occupancy.get(node_id, 0)

    def _move_train(self, train, from_node: int, to_node: int):
        """Update occupancy counter when a train moves."""
        if from_node not in (0, 998, 999):
            self._occupancy[from_node] -= 1
            if self._occupancy[from_node] <= 0:
                del self._occupancy[from_node]
        if to_node not in (0, 998, 999):
            self._occupancy[to_node] = self._occupancy.get(to_node, 0) + 1

    def _is_scheduled_stop(self, train, station_name: str) -> bool:
        """True if this train has a scheduled halt at this station."""
        sched = self.schedule.get(train['id'], {})
        return station_name in sched.get('stops', [])

    def _get_train_km(self, train) -> float:
        return self.get_node_km(train['position'])

    def _get_next_stop_km(self, train) -> float:
        """km of the next scheduled stop ahead of this train."""
        sched   = self.schedule.get(train['id'], {})
        stops   = sched.get('stops', [])
        my_km   = self._get_train_km(train)
        direction = train['direction']

        best_km = SECTION_LENGTH_KM if direction == 'DOWN' else 0.0
        found   = False

        for stop_name in stops:
            if stop_name not in self.station_nodes:
                continue
            stop_km = self.station_nodes[stop_name]['km']
            if direction == 'DOWN' and stop_km > my_km:
                if not found or stop_km < best_km:
                    best_km = stop_km
                    found   = True
            elif direction == 'UP' and stop_km < my_km:
                if not found or stop_km > best_km:
                    best_km = stop_km
                    found   = True

        return best_km

    def _is_in_token_block(self, node_id: int) -> bool:
        return node_id in self.ghat_token.token_block_ids

    def _is_banker_point(self, node_id: int) -> bool:
        node = self.track_map.get(node_id, {})
        return node.get('is_banker_point', False)

    def _station_platform_availability(self, station_name: str) -> float:
        """Free platforms / total platforms at a station. [0, 1]"""
        if station_name not in self.station_nodes:
            return 1.0
        pf_ids = self.station_nodes[station_name]['platforms']
        if not pf_ids:
            return 1.0
        free = sum(
            1 for pid in pf_ids
            if self.get_node_occupancy(pid) < self.track_map[pid].get('capacity', 1)
        )
        return free / len(pf_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL CHECK — km-based, direction-aware
    # ─────────────────────────────────────────────────────────────────────────

    def check_signal(self, train_idx: int):
        """
        Scan ahead (in train's direction of travel) for capacity conflicts.

        Returns
        -------
        signal_value : float  0.0=clear, 1.0=caution, 2.0=danger
        dist_km      : float  km to nearest danger
        """
        train     = self.trains[train_idx]
        current   = train['position']
        direction = train['direction']

        if current in (0, 998, 999):
            return 0.0, DANGER_HORIZON_KM

        danger_val = 0.0
        dist_km    = DANGER_HORIZON_KM
        my_km      = self.get_node_km(current)

        # Walk forward up to DANGER_HORIZON_KM
        cursor = current
        for _ in range(30):   # max 30 hops (~5km each at finest granularity)
            node_data  = self.track_map.get(cursor, {})
            next_opts  = node_data.get('next', [])
            if not next_opts:
                break

            # For UP trains we walk BACKWARDS through next_opts of the
            # reversed graph — simplified: use km comparison to determine
            # which next node is "ahead" in the UP direction
            if direction == 'UP':
                # UP trains move toward lower km values
                candidates = [n for n in next_opts
                              if self.get_node_km(n) < self.get_node_km(cursor)]
                if not candidates:
                    candidates = next_opts
                target = min(candidates, key=lambda n: self.get_node_km(n))
            else:
                target = next_opts[0]

            target_km  = self.get_node_km(target)
            gap_km     = abs(target_km - my_km)

            if gap_km > DANGER_HORIZON_KM:
                break

            occ = self.get_node_occupancy(target)
            cap = self.track_map.get(target, {}).get('capacity', 1)

            # Token block: treat as danger if opposing direction holds token
            if self._is_in_token_block(target):
                tok = self.ghat_token.status()
                if tok['direction'] is not None and tok['direction'] != direction:
                    if danger_val < 2.0:
                        danger_val = 2.0
                        dist_km    = gap_km
                    break

            if occ >= cap:
                if danger_val < 2.0:
                    danger_val = 2.0
                    dist_km    = gap_km
                break
            elif occ >= cap - 1 and danger_val < 1.0:
                danger_val = 1.0
                dist_km    = gap_km

            cursor = target

        return danger_val, dist_km

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION MASK
    # ─────────────────────────────────────────────────────────────────────────

    def get_action_mask(self) -> np.ndarray:
        """
        Returns bool array (MAX_TRAINS_CAPACITY, 3).
        Actions: 0=HOLD, 1=PROCEED_MAIN, 2=DIVERT_LOOP

        Masking rules (in priority order):
          1. Ghost train slots — only HOLD allowed
          2. Finished trains — only HOLD allowed
          3. Not yet spawned — only HOLD
          4. Dwell time remaining — only HOLD
          5. No next nodes — only HOLD
          6. Token block: opposing direction holds token — block PROCEED
          7. Banker wait: train is attaching/detaching banker — only HOLD
          8. Capacity check on main target — block PROCEED if full
          9. Capacity check on loop targets — block DIVERT if all full
        """
        mask = np.zeros((MAX_TRAINS_CAPACITY, 3), dtype=bool)

        for i in range(MAX_TRAINS_CAPACITY):
            # Default: HOLD always legal
            mask[i, 0] = True

            # Ghost slot
            if i >= len(self.trains):
                continue

            train = self.trains[i]

            # Finished
            if train['finished']:
                continue

            current_pos = train['position']
            direction   = train['direction']

            # Not yet spawned
            if current_pos in (0, 998):
                continue

            # Banker attach/detach in progress
            if train.get('banker_wait', 0) > 0:
                continue

            # Dwell time remaining
            if train.get('dwell_rem', 0) > 0:
                continue

            node_data = self.track_map.get(current_pos, {})
            next_opts = node_data.get('next', [])
            if not next_opts:
                continue

            # Determine main target (first in next_opts for DOWN,
            # lowest-km next node for UP)
            if direction == 'UP':
                main_candidates = [n for n in next_opts
                                   if self.get_node_km(n) <= self.get_node_km(current_pos)]
                main_target = (min(main_candidates, key=lambda n: self.get_node_km(n))
                               if main_candidates else next_opts[0])
            else:
                main_target = next_opts[0]

            # Token block check — applies before capacity check
            if self._is_in_token_block(main_target):
                if not self.ghat_token.can_enter(train['id'], direction):
                    # Opposing train holds mid-line — force HOLD or DIVERT
                    # PROCEED blocked entirely
                    pass
                else:
                    # Can enter token block
                    main_occ = self.get_node_occupancy(main_target)
                    main_cap = self.track_map.get(main_target, {}).get('capacity', 1)
                    if main_occ < main_cap:
                        mask[i, 1] = True
            else:
                # Normal capacity check for main target
                main_occ = self.get_node_occupancy(main_target)
                main_cap = self.track_map.get(main_target, {}).get('capacity', 1)
                if main_occ < main_cap:
                    mask[i, 1] = True

            # DIVERT check — loop/platform nodes (next_opts[1:])
            loop_targets = [n for n in next_opts if n != main_target]
            for loop_n in loop_targets:
                occ = self.get_node_occupancy(loop_n)
                cap = self.track_map.get(loop_n, {}).get('capacity', 1)
                if occ < cap:
                    mask[i, 2] = True
                    break

        # OR-solver overlay (optional)
        shield = self._feasibility_shield
        if shield is not None:
            mask = shield.get_masked_actions(self.sim_time, mask)

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((MAX_TRAINS_CAPACITY, N_FEATURES), dtype=np.float32)

        # Section-level stats (computed once per step)
        active_trains  = [t for t in self.trains if not t['finished']
                          and t['position'] not in (0, 998)]
        section_load   = len(active_trains) / max(MAX_TRAINS_CAPACITY, 1)

        tok_status_global = self.ghat_token.status()

        for i in range(MAX_TRAINS_CAPACITY):
            if i >= len(self.trains):
                obs[i] = _GHOST_OBS
                continue

            train     = self.trains[i]
            pos       = train['position']
            direction = train['direction']
            sched     = self.schedule.get(train['id'], {})

            if train['finished']:
                obs[i] = _GHOST_OBS
                continue

            my_km       = self.get_node_km(pos)
            dir_norm    = 1.0 if direction == 'UP' else 0.0

            # Route progress: km done toward destination
            if direction == 'DOWN':
                km_done = my_km
            else:
                km_done = SECTION_LENGTH_KM - my_km
            route_progress = km_done / SECTION_LENGTH_KM

            # Current delay
            current_delay = max(0, self.sim_time - sched.get('deadline', 9999))
            delay_norm    = min(current_delay, 60) / 60.0

            # Dwell
            dwell_norm = (train.get('dwell_rem', 0) /
                          max(DWELL_TIME_PLATFORM, 1))

            # Signal
            sig_val, sig_dist_km = self.check_signal(i)
            sig_norm   = sig_val / 2.0
            danger_norm = min(sig_dist_km, DANGER_HORIZON_KM) / DANGER_HORIZON_KM

            # Lead train (same direction, ahead)
            dist_to_lead = SPACING_HORIZON_KM
            lead_speed   = 0.0
            lead_prio    = 0.0
            highest_prio_ahead = 0.0

            for j, other in enumerate(self.trains):
                if i == j or other['finished']:
                    continue
                if other['direction'] != direction:
                    continue
                other_km = self.get_node_km(other['position'])
                if direction == 'DOWN' and other_km > my_km:
                    gap = other_km - my_km
                    if gap < dist_to_lead:
                        dist_to_lead = gap
                        lead_speed   = other['speed']
                        lead_prio    = other['priority']
                elif direction == 'UP' and other_km < my_km:
                    gap = my_km - other_km
                    if gap < dist_to_lead:
                        dist_to_lead = gap
                        lead_speed   = other['speed']
                        lead_prio    = other['priority']
                if other['priority'] > highest_prio_ahead:
                    highest_prio_ahead = other['priority']

            lead_dist_norm  = min(dist_to_lead, SPACING_HORIZON_KM) / SPACING_HORIZON_KM
            lead_speed_norm = lead_speed / MAX_SPEED
            lead_prio_norm  = lead_prio / 6.0

            # Opposing train (different direction, anywhere on section)
            opp_present  = 0.0
            opp_dist     = SECTION_LENGTH_KM
            opp_speed    = 0.0
            opp_prio     = 0.0

            for j, other in enumerate(self.trains):
                if i == j or other['finished']:
                    continue
                if other['direction'] == direction:
                    continue
                if other['position'] in (0, 998, 999):
                    continue
                other_km = self.get_node_km(other['position'])
                gap      = abs(my_km - other_km)
                if gap < opp_dist:
                    opp_dist  = gap
                    opp_speed = other['speed']
                    opp_prio  = other['priority']
                    opp_present = 1.0

            opp_dist_norm  = min(opp_dist, SECTION_LENGTH_KM) / SECTION_LENGTH_KM
            opp_speed_norm = opp_speed / MAX_SPEED
            opp_prio_norm  = opp_prio / 6.0

            # Token status from this train's perspective
            tok = tok_status_global
            if not tok['is_free']:
                if tok['direction'] == direction:
                    token_feat = 0.5   # same direction holds — we might join
                else:
                    token_feat = 1.0   # opposing holds — we're blocked
            else:
                token_feat = 0.0       # free

            # Next scheduled stop distance
            next_stop_km   = self._get_next_stop_km(train)
            stop_dist_km   = abs(next_stop_km - my_km)
            stop_dist_norm = min(stop_dist_km, SECTION_LENGTH_KM) / SECTION_LENGTH_KM

            # Banker point awareness
            at_banker = 1.0 if self._is_banker_point(pos) else 0.0
            banker_wait_norm = min(
                train.get('banker_wait', 0), BANKER_ATTACH_TIME
            ) / max(BANKER_ATTACH_TIME, 1)

            # Platform availability at next stop
            # Find station name for next stop km
            next_stop_station = None
            for sname, sdata in self.station_nodes.items():
                if abs(sdata['km'] - next_stop_km) < 1.0:
                    next_stop_station = sname
                    break
            plat_avail = (self._station_platform_availability(next_stop_station)
                          if next_stop_station else 1.0)

            # Same-direction train count
            same_dir = sum(1 for t in self.trains
                           if not t['finished']
                           and t['direction'] == direction
                           and t['position'] not in (0, 998, 999))
            same_dir_norm = same_dir / max(MAX_TRAINS_CAPACITY, 1)

            # Deadline remaining
            deadline_rem  = max(0, sched.get('deadline', 0) - self.sim_time)
            deadline_norm = min(deadline_rem, 500) / 500.0

            # Assemble 23 features
            obs[i, 0]  = train['speed'] / MAX_SPEED          # my_speed
            obs[i, 1]  = train['priority'] / 6.0             # my_priority
            obs[i, 2]  = dir_norm                            # direction
            obs[i, 3]  = route_progress                      # route_progress
            obs[i, 4]  = delay_norm                          # delay_norm
            obs[i, 5]  = dwell_norm                          # dwell_norm
            obs[i, 6]  = sig_norm                            # signal_value
            obs[i, 7]  = danger_norm                         # dist_to_danger
            obs[i, 8]  = lead_dist_norm                      # dist_to_lead
            obs[i, 9]  = lead_speed_norm                     # lead_speed
            obs[i, 10] = opp_present                         # opposing_present
            obs[i, 11] = opp_dist_norm                       # opposing_dist
            obs[i, 12] = opp_speed_norm                      # opposing_speed
            obs[i, 13] = opp_prio_norm                       # opposing_priority
            obs[i, 14] = token_feat                          # token_status
            obs[i, 15] = stop_dist_norm                      # dist_to_next_stop
            obs[i, 16] = at_banker                           # at_banker_point
            obs[i, 17] = banker_wait_norm                    # banker_wait
            obs[i, 18] = plat_avail                          # platform_avail
            obs[i, 19] = section_load                        # section_load
            obs[i, 20] = same_dir_norm                       # same_dir_trains
            obs[i, 21] = deadline_norm                       # deadline_norm
            obs[i, 22] = highest_prio_ahead / 6.0            # lead_priority

        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        self.sim_time += 1
        reward      = 0.0
        terminated  = False

        # Traffic tax — penalise congestion per active train
        num_active = sum(1 for t in self.trains
                         if not t['finished'] and t['position'] not in (0, 998))
        reward -= 0.005 * num_active

        # Process highest-priority trains first (they claim capacity first)
        sorted_idx = sorted(
            range(len(self.trains)),
            key=lambda k: self.trains[k].get('priority', 0),
            reverse=True,
        )

        current_positions = []

        for i in sorted_idx:
            train    = self.trains[i]
            sched    = self.schedule[train['id']]
            act      = int(np.clip(action[i], 0, 2))
            pos      = train['position']
            direction = train['direction']

            # ── Already finished ──────────────────────────────────────────
            if train['finished']:
                current_positions.append(999)
                continue

            # ── Not yet spawned ───────────────────────────────────────────
            if self.sim_time < sched['start_time']:
                current_positions.append(pos)
                continue

            # ── Spawn: move from staging node to first real node ──────────
            if pos in (0, 998):
                # DOWN trains enter at CSMT switch_in (node 1)
                # UP trains enter at Manmad switch_in
                if direction == 'DOWN':
                    entry_node = self.station_nodes['CSMT']['switch_in']
                else:
                    entry_node = self.station_nodes['MANMAD']['switch_in']

                entry_cap = self.track_map.get(entry_node, {}).get('capacity', 2)
                if self.get_node_occupancy(entry_node) >= entry_cap:
                    reward -= 0.005   # blocked at entry
                    current_positions.append(pos)
                    continue

                self._move_train(train, pos, entry_node)
                train['position'] = entry_node
                pos = entry_node
                self._train_speeds[i] = 0.0

            node_data  = self.track_map.get(pos, {})
            track_limit = node_data.get('speed', train['max_speed'])

            # ── Banker attach/detach wait ─────────────────────────────────
            if train.get('banker_wait', 0) > 0:
                train['banker_wait'] -= 1
                if train['banker_wait'] == 0:
                    # Attach complete — train can now enter token block
                    if direction == 'UP':
                        train['banker_attached'] = True
                    else:
                        # Detach complete (DOWN train leaving ghat)
                        train['banker_attached'] = False
                current_positions.append(pos)
                continue

            # ── Dwell time ────────────────────────────────────────────────
            if train.get('dwell_rem', 0) > 0:
                train['dwell_rem'] -= 1
                train['target_speed'] = 0
                self._train_speeds[i] = 0.0
                act = 0   # force HOLD during dwell

            # ── Target speed ──────────────────────────────────────────────
            # Ghat speed override
            if node_data.get('token_block') and node_data.get('gradient'):
                track_limit = (50 if direction == 'UP' else 60)

            train['target_speed'] = (
                min(track_limit, train['max_speed']) if act in (1, 2) else 0
            )

            # ── Braking override ──────────────────────────────────────────
            sig_val, sig_dist_km = self.check_signal(i)
            if train['speed'] > 0 and sig_val == 2.0:
                d_brake = (train['speed'] ** 2) / (2.0 * train['decel_rate'] * 60.0)
                if sig_dist_km <= d_brake:
                    train['target_speed'] = 0

            # ── Speed inertia ─────────────────────────────────────────────
            if train['target_speed'] > train['speed']:
                train['speed'] = min(train['target_speed'],
                                     train['speed'] + train['accel_rate'])
            elif train['target_speed'] < train['speed']:
                train['speed'] = max(train['target_speed'],
                                     train['speed'] - train['decel_rate'])
            train['speed'] = max(0, min(train['speed'],
                                        min(track_limit, train['max_speed'])))
            self._train_speeds[i] = train['speed']

            # ── Movement accumulator ──────────────────────────────────────
            self._movement_acc[i] += train['speed'] / 60.0

            # ── HOLD ──────────────────────────────────────────────────────
            if act == 0:
                node_type = node_data.get('type', '')
                if 'MAIN' in node_type or 'GHAT' in node_type:
                    if sig_val == 0.0 and train['speed'] == 0:
                        reward -= 0.02   # idle on main line — bad
                else:
                    if train.get('dwell_rem', 0) == 0:
                        reward -= 0.01   # loitering in loop after dwell done
                    else:
                        reward -= 0.0005
                current_positions.append(pos)

            # ── PROCEED or DIVERT ─────────────────────────────────────────
            else:
                if self._movement_acc[i] < 1.0:
                    # Hasn't moved a full block yet
                    current_positions.append(pos)
                    continue

                next_opts = node_data.get('next', [])
                if not next_opts:
                    reward -= 0.01
                    current_positions.append(pos)
                    continue

                # Resolve main target (direction-aware)
                if direction == 'UP':
                    main_candidates = [
                        n for n in next_opts
                        if self.get_node_km(n) <= self.get_node_km(pos)
                    ]
                    main_target = (
                        min(main_candidates, key=lambda n: self.get_node_km(n))
                        if main_candidates else next_opts[0]
                    )
                else:
                    main_target = next_opts[0]

                loop_targets = [n for n in next_opts if n != main_target]

                # ── DIVERT (act == 2) ──────────────────────────────────────
                if act == 2 and loop_targets:
                    target_node = None
                    for lnode in loop_targets:
                        if self.get_node_occupancy(lnode) < self.track_map.get(lnode, {}).get('capacity', 1):
                            target_node = lnode
                            break

                    if target_node is None:
                        # All loops full — fall back to main
                        target_node = main_target
                        reward -= 0.02
                    else:
                        # Successful divert — apply dwell if scheduled stop
                        lnode_station = self.track_map.get(target_node, {}).get('station')
                        if (lnode_station and
                                self._is_scheduled_stop(train, lnode_station) and
                                self.track_map[target_node]['type'] == 'PLATFORM'):
                            train['dwell_rem'] = DWELL_TIME_PLATFORM
                        elif self.track_map.get(target_node, {}).get('type') == 'LOOP':
                            train['dwell_rem'] = DWELL_TIME_LOOP

                        # Reward: low-priority train yielding is good
                        if train['priority'] < 5:
                            reward += 0.05
                else:
                    target_node = main_target

                # ── Token block entry check ───────────────────────────────
                if self._is_in_token_block(target_node):
                    if not self.ghat_token.can_enter(train['id'], direction):
                        # Token held by opposing direction — hard block
                        train['speed'] = 0
                        train['target_speed'] = 0
                        self._train_speeds[i] = 0
                        reward -= 0.05
                        current_positions.append(pos)
                        continue

                # ── Capacity check at commit time ─────────────────────────
                commit_occ = self.get_node_occupancy(target_node)
                commit_cap = self.track_map.get(target_node, {}).get('capacity', 1)
                if commit_occ >= commit_cap:
                    train['speed'] = 0
                    train['target_speed'] = 0
                    self._train_speeds[i] = 0
                    reward -= 0.05
                    current_positions.append(pos)
                    continue

                # ── Commit move ───────────────────────────────────────────
                old_pos = pos
                self._move_train(train, old_pos, target_node)
                train['position'] = target_node
                self._movement_acc[i] -= 1.0

                # Token system update
                was_in_token = self._is_in_token_block(old_pos)
                now_in_token = self._is_in_token_block(target_node)

                if now_in_token and not was_in_token:
                    self.ghat_token.train_entered(train['id'], direction)
                elif was_in_token and not now_in_token:
                    self.ghat_token.train_exited(train['id'])

                    # Banker detach check: UP train exiting ghat at Igatpuri
                    igatpuri_data = self.station_nodes.get('IGATPURI', {})
                    if (direction == 'UP' and
                            train.get('banker_required') and
                            target_node == igatpuri_data.get('switch_in')):
                        train['banker_wait'] = BANKER_DETACH_TIME

                    # DOWN train entering ghat needs banker detach at Kasara exit
                    kasara_data = self.station_nodes.get('KASARA', {})
                    if (direction == 'DOWN' and
                            train.get('banker_required') and
                            old_pos == kasara_data.get('switch_out')):
                        # Banker attaches at Kasara for DOWN trains before entering ghat
                        train['banker_wait'] = BANKER_ATTACH_TIME

                # Banker attach: UP train arriving at Kasara
                kasara_sw_in = self.station_nodes.get('KASARA', {}).get('switch_in')
                if (direction == 'UP' and
                        train.get('banker_required') and
                        not train.get('banker_attached') and
                        target_node == kasara_sw_in):
                    train['banker_wait'] = BANKER_ATTACH_TIME

                # Station pass-through dwell logic
                target_station = self.track_map.get(target_node, {}).get('station')
                if (target_station and
                        self.track_map[target_node]['type'] == 'PLATFORM' and
                        self._is_scheduled_stop(train, target_station)):
                    train['dwell_rem'] = DWELL_TIME_PLATFORM

                # Travel efficiency reward
                if target_node not in train['visited_nodes']:
                    train['visited_nodes'].add(target_node)
                    eff = train['speed'] / max(track_limit, 1)
                    reward += eff * (train['priority'] / 6.0) * 0.05
                    reward += 0.3 * (train['priority'] / 6.0)
                else:
                    reward -= 0.005   # revisiting — loop escape failed

                # Destination check
                dest_node = (999 if direction == 'DOWN'
                             else self.station_nodes['CSMT']['switch_out'])
                if target_node == dest_node or not self.track_map.get(target_node, {}).get('next'):
                    train['position'] = 999
                    train['finished']  = True
                    train['speed']     = 0
                    self._move_train(train, target_node, 999)
                    reward += 30.0
                    current_positions.append(999)
                    continue

                current_positions.append(target_node)

            # ── Punctuality penalty ───────────────────────────────────────
            delay = max(0, self.sim_time - sched['deadline'])
            if delay > 0:
                reward -= delay * (train['priority'] / 6.0) * 0.005
                reward -= (delay ** 2) * (train['priority'] / 6.0) * 0.0005

        # ── Collision detection ───────────────────────────────────────────
        pos_counts = Counter(
            p for p in current_positions if p not in (0, 998, 999)
        )
        for node, count in pos_counts.items():
            cap = self.track_map.get(node, {}).get('capacity', 1)
            if count > cap:
                crashers = [t['id'] for t in self.trains if t['position'] == node]
                print(f"\n{'X'*50}")
                print(f"💥 COLLISION @ step {self.sim_time} node {node}")
                print(f"   Trains: {crashers}")
                print(f"{'X'*50}\n")
                reward    -= 75.0
                terminated = True

        # ── Episode termination ───────────────────────────────────────────
        # All spawned trains finished
        spawned = [p for p in current_positions if p != 0]
        if spawned and all(p == 999 for p in spawned):
            reward    += 40.0
            terminated = True

        # Timeout
        last_spawn      = max(s['start_time'] for s in self.schedule.values())
        max_allowed     = last_spawn + 1500
        if self.sim_time > max_allowed:
            _log.warning(f"[TIMEOUT] sim_time={self.sim_time} max={max_allowed}")
            terminated = True

        return self._get_observation(), reward, terminated, False, {}

    # ─────────────────────────────────────────────────────────────────────────
    # CHAOS MONKEY
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_chaos(self):
        rng = np.random.default_rng()
        for train in self.trains:
            if rng.random() < self.chaos_delay_prob:
                mins = int(rng.integers(self.chaos_delay_min, self.chaos_delay_max + 1))
                tid  = train['id']
                if tid in self.schedule:
                    self.schedule[tid]['start_time'] += mins
                    self.schedule[tid]['deadline']   += mins
                train['delay'] = mins

        if self.chaos_speed_snag and self.trains:
            victim = rng.choice(self.trains)
            victim['max_speed'] = max(1, int(victim['max_speed'] * self.chaos_speed_factor))