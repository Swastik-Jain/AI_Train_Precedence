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

Fix applied (2026-05-21):
  - Each train dict now carries 'finish_step' (int | None), set to the
    exact sim_time step when the train reaches its destination node.
    Previously all delay/on-time metrics used episode end time (ep_len)
    for every train, inflating delays for early finishers and washing
    out real dispatcher-quality differences. benchmark.py and the two
    baseline scripts now read finish_step instead of ep_len when
    computing per-train delay.
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
# OBSERVATION SPACE — 25 features per train
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
# Category 7 — Urgency (1 feature)
#   23 required_speed_norm  (dist_remaining_km / steps_remaining) / MAX_SPEED
#                           = how fast the train MUST move to make deadline.
#                           > 1.0 means impossible; 0 means already late/finished.
#
# Category 8 — Divert awareness (1 feature)  ← NEW (Phase 4)
#   24 nearest_loop_dist_norm  km to nearest available loop or crossing loop
#                              ahead in this train's direction / SECTION_LENGTH_KM.
#                              0.0 = loop right here, 1.0 = none available.
#                              Covers station loops (AMBERNATH/TITWALA/ATGAON etc.)
#                              AND mid-section crossing loops (Nandgaon/Lasalgaon).
#
N_FEATURES = 25

# Ghost train padding — represents an empty slot
_GHOST_OBS = np.zeros(N_FEATURES, dtype=np.float32)
_GHOST_OBS[7]  = 1.0   # dist_to_danger: far
_GHOST_OBS[8]  = 1.0   # dist_to_lead: far
_GHOST_OBS[11] = 1.0   # opposing_dist: far
_GHOST_OBS[23] = 0.0   # required_speed: ghost slot has no urgency
_GHOST_OBS[24] = 1.0   # nearest_loop_dist: no loop in range


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

        # Schedule generation mode ('normal' or 'stress')
        self._schedule_mode     = 'normal'
        self._num_trains        = len(ACTIVE_FLEET)

        self.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def set_difficulty(self, num_trains: int):
        self._schedule_mode = 'normal'
        self._num_trains    = num_trains
        fleet, sched = generate_daily_schedule(num_trains)
        self.active_fleet = fleet
        self.schedule     = sched
        self.reset()
        print(f"🔹 Difficulty set: {num_trains} trains")

    def set_stress_mode(self, num_trains: int):
        """Use clustered-spawn stress schedule for evaluation benchmarks."""
        from ai.config import generate_stress_schedule
        self._schedule_mode = 'stress'
        self._num_trains    = num_trains
        fleet, sched = generate_stress_schedule(num_trains)
        self.active_fleet = fleet
        self.schedule     = sched
        self.reset()
        print(f"🔥 Stress mode: {num_trains} trains (clustered spawns)")

    def set_mixed_mode(self, num_trains: int):
        """Use 40% stress + 60% normal schedules for training."""
        self._schedule_mode = 'mixed'
        self._num_trains    = num_trains
        self.reset()
        print(f"🔀 Mixed mode: {num_trains} trains (40% stress)")

    def set_custom_schedule(self, fleet: list, schedule: dict):
        """Allows inference runner to inject an exact schedule without regenerating it."""
        self._schedule_mode = 'custom'
        self.active_fleet = []
        for t in fleet:
            train = t.copy()
            # Ensure standard RL state fields
            train.setdefault('speed', 0)
            train.setdefault('target_speed', 0)
            train.setdefault('delay', 0)
            train.setdefault('idle_time', 0)
            train.setdefault('dwell_rem', 0)
            train.setdefault('finished', False)
            train['finish_step'] = None
            train.setdefault('banker_attached', False)
            train.setdefault('banker_wait', 0)
            train['visited_nodes'] = set()
            # Use sensible defaults for physics parameters if missing
            train.setdefault('accel_rate', 10)
            train.setdefault('decel_rate', 15)
            self.active_fleet.append(train)
        self.schedule = {k: v.copy() for k, v in schedule.items()}
        self._num_trains = len(fleet)

    def attach_feasibility_shield(self, shield):
        self._feasibility_shield = shield
        status = "attached" if shield is not None else "detached"
        print(f"🛡️  FeasibilityShield {status}.")

    def set_chaos_mode(self, enabled=True, hardcore=False):
        self.chaos_mode         = enabled
        self.chaos_delay_prob   = 0.60 if hardcore else 0.30
        self.chaos_delay_min    = 5 if hardcore else 1
        self.chaos_delay_max    = 30 if hardcore else 10
        self.chaos_speed_snag   = True
        self.chaos_speed_factor = 0.50 if hardcore else 0.80
        state = "HARDCORE 👹" if hardcore else ("ENABLED 🐒" if enabled else "DISABLED ✅")
        print(f"⚡ Chaos Monkey {state} | delay_prob={self.chaos_delay_prob:.0%} "
              f"delay={self.chaos_delay_min}–{self.chaos_delay_max}m speed_snag={self.chaos_speed_snag}({self.chaos_speed_factor:.0%})")

    def set_incident_mode(self, enabled=True):
        self.incident_mode = enabled
        state = "ENABLED 🚨" if enabled else "DISABLED ✅"
        print(f"🚨 Incident Mode {state}")

    def apply_incident(self):
        """Randomly selects a main track node and breaks it (capacity = 0)."""
        main_nodes = [nid for nid, data in self.track_map.items() if data.get('type') == 'MAIN_TRACK']
        if main_nodes:
            incident_node = self.np_random.choice(main_nodes)
            self.track_map[incident_node]['capacity'] = 0
            self.track_map[incident_node]['status'] = 'BROKEN'
            print(f"🚨 INCIDENT: Node {incident_node} has permanently broken down!")

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim_time = 0

        # Fresh fleet + schedule each episode (randomized)
        if self._schedule_mode == 'mixed':
            from ai.config import generate_stress_schedule
            # 40% stress, 60% normal — teaches agent both scenarios
            if self.np_random.random() < 0.4:
                self.active_fleet, self.schedule = generate_stress_schedule(
                    self._num_trains
                )
            else:
                self.active_fleet, self.schedule = generate_daily_schedule(
                    self._num_trains
                )
        elif self._schedule_mode == 'stress':
            from ai.config import generate_stress_schedule
            self.active_fleet, self.schedule = generate_stress_schedule(
                self._num_trains
            )
        elif self._schedule_mode == 'custom':
            pass # Keep self.active_fleet and self.schedule exactly as injected
        else:
            self.active_fleet, self.schedule = generate_daily_schedule(
                self._num_trains
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
            train['finish_step']     = None   # set to sim_time when train reaches destination
            train['committed_next_node'] = None  # which node's edge this train is currently on / about to enter — set every step(), read by simulation_service.py for the visual edge mapping
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
        # Node 998: UP trains wait here before entering Manmad switch_out
        manmad_switch_out = self.station_nodes['MANMAD']['switch_out']
        self.track_map[998] = {
            'type':        'ORIGIN',
            'speed':       0,
            'capacity':    99,
            'prev':        [manmad_switch_out],
            'next':        [],
            'km':          SECTION_LENGTH_KM,
            'station':     None,
            'token_block': False,
            'gradient':    False,
        }

        # Token system for Kasara-Igatpuri ghat
        self.ghat_token = GhatTokenSystem(self.token_blocks)

        if getattr(self, 'incident_mode', False):
            self.apply_incident()

        # O(1) occupancy — rebuild from current positions
        self.sim_time = 0
        self._occupancy = {}  # format: {node_id: {'UP': count, 'DOWN': count}}
        self._movement_acc = np.zeros(MAX_TRAINS_CAPACITY, dtype=np.float32)
        self._train_speeds = np.zeros(MAX_TRAINS_CAPACITY, dtype=np.float32)

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

    def get_node_occupancy(self, node_id: int, direction: str = None) -> int:
        """O(1) occupancy lookup. If direction is specified, returns occupancy for that specific direction."""
        occ = self._occupancy.get(node_id, {})
        if direction:
            return occ.get(direction, 0)
        return sum(occ.values())

    def _move_train(self, train, from_node: int, to_node: int):
        """Update occupancy counter when a train moves."""
        direction = train.get('direction', 'DOWN')
        
        if from_node not in (0, 998, 999):
            if from_node in self._occupancy and direction in self._occupancy[from_node]:
                self._occupancy[from_node][direction] -= 1
                if self._occupancy[from_node][direction] <= 0:
                    del self._occupancy[from_node][direction]
                if not self._occupancy[from_node]:
                    del self._occupancy[from_node]
                    
        if to_node not in (0, 998, 999):
            if to_node not in self._occupancy:
                self._occupancy[to_node] = {}
            self._occupancy[to_node][direction] = self._occupancy[to_node].get(direction, 0) + 1

    def _is_scheduled_stop(self, train, station_name: str) -> bool:
        """True if this train has a scheduled halt at this station."""
        sched = self.schedule.get(train['id'], {})
        return station_name in sched.get('stops', []) or station_name in sched

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

    def _get_nearest_loop_dist_km(self, train) -> float:
        """
        Distance (km) to the nearest available loop or crossing loop ahead
        of this train in its direction of travel.

        Checks two classes of divert nodes:
          1. Station loops  — LOOP-type nodes attached to station switch_in nodes
             (covers AMBERNATH, TITWALA, ATGAON, Kasara, Igatpuri etc.)
          2. Mid-section crossing loops — CROSSING_LOOP-type nodes with station=None
             (covers Nandgaon km210, Lasalgaon km235)

        Returns SECTION_LENGTH_KM when no free loop exists ahead.
        """
        my_km     = self._get_train_km(train)
        direction = train['direction']
        best_dist = SECTION_LENGTH_KM

        # ── Station loops ─────────────────────────────────────────────────
        for sname, sdata in self.station_nodes.items():
            s_km = sdata['km']
            if direction == 'DOWN' and s_km <= my_km:
                continue
            if direction == 'UP' and s_km >= my_km:
                continue
            for lid in sdata.get('loops', []):
                occ = self.get_node_occupancy(lid)
                cap = self.track_map.get(lid, {}).get('capacity', 1)
                if occ < cap:
                    dist = abs(s_km - my_km)
                    if dist < best_dist:
                        best_dist = dist
                    break   # one free loop at this station is enough

        # ── Mid-section crossing loops ─────────────────────────────────────
        for nid, nd in self.track_map.items():
            if nd.get('type') == 'CROSSING_LOOP' and nd.get('station') is None:
                cl_km = nd.get('km', 0)
                if direction == 'DOWN' and cl_km <= my_km:
                    continue
                if direction == 'UP' and cl_km >= my_km:
                    continue
                occ = self.get_node_occupancy(nid)
                cap = nd.get('capacity', 1)
                if occ < cap:
                    dist = abs(cl_km - my_km)
                    if dist < best_dist:
                        best_dist = dist

        return best_dist

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

    def _is_chokepoint_node(self, node_id: int) -> bool:
        """Return True if this is a SWITCH node bordering a token-block section."""
        node_data = self.track_map.get(node_id, {})
        if node_data.get('type') != 'SWITCH':
            return False
        for nxt in node_data.get('next', []):
            if self._is_in_token_block(nxt):
                return True
        for prv in node_data.get('prev', []):
            if self._is_in_token_block(prv):
                return True
        return False

    def _next_section_has_room(self, candidate_node: int, direction: str, directional_check: bool, train_id: str, actual_target: int = None) -> bool:
        """
        Check if the immediate next section after candidate_node has room.
        Used at chokepoint nodes to prevent admitting a train that would instantly get stuck.
        """
        node_data = self.track_map.get(candidate_node, {})
        next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
        
        if not next_opts:
            return True
            
        next_hop = actual_target if actual_target is not None else next_opts[0]
        
        # Check token block ownership for the next hop
        if self._is_in_token_block(next_hop):
            if self.ghat_token and not self.ghat_token.can_enter(train_id, direction):
                return False
                
        cap = self.track_map.get(next_hop, {}).get('capacity', 1)
        
        if directional_check:
            dir_cap = max(1, cap // 2) if cap > 1 else cap
            occ = self.get_node_occupancy(next_hop, direction)
            return occ < dir_cap
        else:
            occ = self.get_node_occupancy(next_hop)
            return occ < cap

    def is_safe_to_proceed(self, train: dict, target_node: int, current_occ: dict = None) -> bool:
        """
        Unified safety check consulted by both the RL mask and OR-Shield (anti-loiter).
        Checks token block availability, capacity, and chokepoint look-ahead.
        """
        direction = train['direction']
        
        # 1. Chokepoint Look-ahead
        if self._is_chokepoint_node(target_node):
            if not self._next_section_has_room(target_node, direction, directional_check=True, train_id=train['id']):
                return False
                
        # 2. Token Block Ownership
        if self._is_in_token_block(target_node):
            if self.ghat_token and not self.ghat_token.can_enter(train['id'], direction):
                return False
                
        # 3. Capacity Check
        main_cap = self.track_map.get(target_node, {}).get('capacity', 1)
        dir_cap = max(1, main_cap // 2) if main_cap > 1 else main_cap
        
        if current_occ is not None:
            # current_occ is expected to be a nested dict: {node: {direction: count}}
            occ = current_occ.get(target_node, {}).get(direction, 0)
        else:
            occ = self.get_node_occupancy(target_node, direction)
            
        if occ >= dir_cap:
            return False
            
        return True

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
            next_opts  = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
            if not next_opts:
                break

            # Look for an available path instead of blindly taking the first one
            target = None
            for opt in next_opts:
                cap = self.track_map.get(opt, {}).get('capacity', 1)
                dir_cap = max(1, cap // 2) if cap > 1 else cap
                occ = self.get_node_occupancy(opt, direction)
                # If the path is a token block, check if opposing holds it
                if self._is_in_token_block(opt):
                    tok = self.ghat_token.status()
                    if tok['direction'] is not None and tok['direction'] != direction:
                        continue # Opposing holds token, this path is blocked
                if occ < dir_cap:
                    target = opt
                    break
            
            if target is None:
                # All options occupied, default to first to trigger danger logic
                target = next_opts[0]

            target_km  = self.get_node_km(target)
            gap_km     = abs(target_km - my_km)

            if gap_km > DANGER_HORIZON_KM:
                break

            cap = self.track_map.get(target, {}).get('capacity', 1)
            dir_cap = max(1, cap // 2) if cap > 1 else cap
            occ = self.get_node_occupancy(target, direction)

            # Token block: treat as danger if opposing direction holds token
            if self._is_in_token_block(target):
                tok = self.ghat_token.status()
                if tok['direction'] is not None and tok['direction'] != direction:
                    if danger_val < 2.0:
                        danger_val = 2.0
                        dist_km    = gap_km
                    break

            if occ >= dir_cap:
                if danger_val < 2.0:
                    danger_val = 2.0
                    dist_km    = gap_km
                break
            elif occ >= dir_cap - 1 and danger_val < 1.0:
                danger_val = 1.0
                dist_km    = gap_km

            cursor = target

        return danger_val, dist_km

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION MASK
    # ─────────────────────────────────────────────────────────────────────────

    def _get_valid_loops(self, loop_targets, direction):
        import math
        scored = []
        dadar_fallback = []

        for lnode in loop_targets:
            lnode_data = self.track_map.get(lnode, {})
            node_type = lnode_data.get('type')
            st_name = lnode_data.get('station')

            if node_type == 'PLATFORM':
                type_rank = 0
                idx_key = 'platforms'
                idx = lnode_data.get('platform_index', 0)
            elif node_type in ('LOOP', 'CROSSING_LOOP'):
                type_rank = 1
                idx_key = 'loops'
                idx = lnode_data.get('loop_index', 0)
            else:
                scored.append((0, 0, lnode))
                continue

            if not st_name or st_name not in self.station_nodes:
                scored.append((type_rank, 0, lnode))
                continue

            if st_name in ('CSMT', 'MANMAD'):
                # Terminus: no directional split — natural centre-out order.
                scored.append((type_rank, idx, lnode))
                continue

            group = self.station_nodes[st_name].get(idx_key, [])
            total = len(group)
            mid = math.ceil(total / 2)
            station_mid = (total - 1) / 2.0

            if direction == 'UP' and idx < mid:
                proximity = abs(idx - station_mid)
                scored.append((type_rank, proximity, lnode))
            elif direction == 'DOWN' and idx >= mid:
                proximity = abs(idx - station_mid)
                scored.append((type_rank, proximity, lnode))
            elif (st_name == 'DADAR' and idx_key == 'loops'
                  and direction == 'DOWN' and total == 1):
                # Last-resort only: Dadar's single loop belongs to UP under the
                # normal split. DOWN may borrow it, but must sort after every
                # station's normal DOWN candidates — handled by returning this
                # list separately and appending it at the very end.
                dadar_fallback.append((type_rank, 999, lnode))

        scored.sort(key=lambda t: (t[0], t[1]))
        dadar_fallback.sort(key=lambda t: (t[0], t[1]))
        return [n for _, _, n in scored] + [n for _, _, n in dadar_fallback]

    def _select_divert_target(self, train: dict, loop_targets: list, direction: str):
        """
        Returns the first available valid loop node, or None if all are full/invalid.
        Persists the chosen platform in train['reserved_platform'] until invalidated.
        """
        reserved = train.get('reserved_platform')
        is_mid_transit = (
            train.get('committed_next_node') == reserved
            or train.get('_early_reservation')
        )

        if reserved is not None and reserved in loop_targets:
            cap = self.track_map.get(reserved, {}).get('capacity', 1)
            physical_occ = self.get_node_occupancy(reserved)
            soft_res_other = sum(1 for tid in getattr(self, '_soft_reservations', {}).get(reserved, []) if tid != train['id'])
            total_occ = physical_occ + soft_res_other

            loop_look_ahead_ok = True
            if self._is_chokepoint_node(reserved):
                if not self._next_section_has_room(reserved, direction, directional_check=True, train_id=train['id'], actual_target=reserved):
                    loop_look_ahead_ok = False
            
            if is_mid_transit:
                if loop_look_ahead_ok:
                    return reserved
            elif total_occ < cap and loop_look_ahead_ok:
                return reserved
            else:
                train['reserved_platform'] = None
                if train['id'] in getattr(self, '_soft_reservations', {}).get(reserved, []):
                    self._soft_reservations[reserved].remove(train['id'])

        ordered = self._get_valid_loops(loop_targets, direction)
        for lnode in ordered:
            cap = self.track_map.get(lnode, {}).get('capacity', 1)
            physical_occ = self.get_node_occupancy(lnode)
            soft_res_other = sum(1 for tid in getattr(self, '_soft_reservations', {}).get(lnode, []) if tid != train['id'])
            total_occ = physical_occ + soft_res_other

            loop_look_ahead_ok = True
            if self._is_chokepoint_node(lnode):
                if not self._next_section_has_room(lnode, direction, directional_check=True, train_id=train['id'], actual_target=lnode):
                    loop_look_ahead_ok = False

            if total_occ < cap and loop_look_ahead_ok:
                train['reserved_platform'] = lnode
                if not hasattr(self, '_soft_reservations'):
                    self._soft_reservations = {}
                if lnode not in self._soft_reservations:
                    self._soft_reservations[lnode] = []
                self._soft_reservations[lnode].append(train['id'])
                return lnode

        return None

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
            next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
            if not next_opts:
                continue

            # Determine main target (first in next_opts for both directions now)
            main_target = next_opts[0]

            # A station's entry switch (SWITCH type with more than one next
            # option) fans out to [switch_out, platforms…, loops…]. There is
            # no direct switch-in → switch-out bypass at these nodes: every
            # train — stopping or not — must ride an explicit platform/loop
            # track through the station. So PROCEED_MAIN (act==1) is never
            # offered here; only HOLD and DIVERT are legal, and DIVERT
            # resolves to the correct platform/loop via the same
            # occupancy-aware selection used for scheduled stops. Whether the
            # train actually dwells is decided purely by the schedule
            # (_is_scheduled_stop), not by which action got it there.
            is_branching_entry = node_data.get('type') == 'SWITCH' and len(next_opts) > 1

            if not is_branching_entry:
                look_ahead_ok = True
                if self._is_chokepoint_node(main_target):
                    if not self._next_section_has_room(main_target, direction, directional_check=True, train_id=train['id'], actual_target=main_target):
                        look_ahead_ok = False
                        # Use a low-noise print or just one that we can easily grep
                        if train.get('speed', 0) > 0 and train.get('position') != 0 and train.get('position') != 998:
                             _log.debug(f"[LOOK-AHEAD MASK DENIAL] Train {train['id']} ({direction}) mask Proceed set to False because chokepoint Node {main_target} has no room next.")

                # Token block check — applies before capacity check
                if self._is_in_token_block(main_target):
                    if not self.ghat_token.can_enter(train['id'], direction):
                        # Opposing train holds mid-line — force HOLD or DIVERT
                        # PROCEED blocked entirely
                        pass
                    else:
                        main_cap = self.track_map.get(main_target, {}).get('capacity', 1)
                        dir_cap = max(1, main_cap // 2) if main_cap > 1 else main_cap
                        main_occ = self.get_node_occupancy(main_target, direction)
                        if main_occ < dir_cap and look_ahead_ok:
                            mask[i, 1] = True
                else:
                    # Normal capacity check for main target
                    main_cap = self.track_map.get(main_target, {}).get('capacity', 1)
                    dir_cap = max(1, main_cap // 2) if main_cap > 1 else main_cap
                    main_occ = self.get_node_occupancy(main_target, direction)
                    if main_occ < dir_cap and look_ahead_ok:
                        mask[i, 1] = True

            # DIVERT check — loop/platform nodes (next_opts[1:])
            loop_targets = [n for n in next_opts if n != main_target]
            valid_loops = self._get_valid_loops(loop_targets, direction)
            
            is_scheduled_stop_here = False
            for loop_n in valid_loops:
                station = self.track_map.get(loop_n, {}).get('station')
                if station and self.track_map[loop_n].get('type') == 'PLATFORM' and self._is_scheduled_stop(train, station):
                    is_scheduled_stop_here = True
                    break
                    
            if is_scheduled_stop_here:
                mask[i, 1] = False

            for loop_n in valid_loops:
                cap = self.track_map.get(loop_n, {}).get('capacity', 1)
                occ = self.get_node_occupancy(loop_n)
                loop_look_ahead_ok = True
                if self._is_chokepoint_node(loop_n):
                    if not self._next_section_has_room(loop_n, direction, directional_check=True, train_id=train['id'], actual_target=loop_n):
                        loop_look_ahead_ok = False
                if occ < cap and loop_look_ahead_ok:
                    mask[i, 2] = True
                    break

            # SAFETY RULE: If train is actively inside the token block, it MUST
            # keep attempting to move every tick. Stopping within the critical gradient section
            # causes trailing trains to collide. We remove HOLD from the
            # mask regardless of whether PROCEED/DIVERT are available — if both
            # are capacity-blocked, the step() physics will reject the move and
            # keep the train in place anyway, but the RL model must keep sending
            # MOVE actions so the moment the node ahead clears, the train goes.
            if self._is_in_token_block(current_pos):
                mask[i, 0] = False   # HOLD never legal inside ghat

        # OR-solver overlay (optional — inference only)
        # FeasibilityShield.get_masked_actions() requires trains, schedule,
        # and ghat_token to apply token + capacity + punctuality constraints.
        shield = self._feasibility_shield
        if shield is not None:
            mask = shield.get_masked_actions(
                sim_time=self.sim_time,
                current_mask=mask,
                trains=self.trains,
                schedule=self.schedule,
                ghat_token=self.ghat_token,
            )

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
            # Ghost slot — padding for unused train capacity
            if i >= len(self.trains):
                obs[i] = _GHOST_OBS
                continue

            train     = self.trains[i]
            pos       = train.get('position', 0)
            direction = train.get('direction', 'DOWN')
            sched     = self.schedule.get(train.get('id', ''), {})

            if train.get('finished', False) or self.sim_time < sched.get('start_time', 0):
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
                if self.sim_time < self.schedule.get(other.get('id', ''), {}).get('start_time', 0):
                    continue
                if other['direction'] != direction:
                    continue

                my_node = self.track_map.get(train['position'], {})
                other_node = self.track_map.get(other['position'], {})
                # Parallel track exclusion: different nodes at the same station
                if my_node.get('station') and my_node.get('station') == other_node.get('station'):
                    if train['position'] != other['position']:
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

                my_node = self.track_map.get(train['position'], {})
                other_node = self.track_map.get(other['position'], {})
                # Parallel track exclusion: different nodes at the same station
                if my_node.get('station') and my_node.get('station') == other_node.get('station'):
                    if train['position'] != other['position']:
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

            # Required speed — urgency feature (Fix C / Phase 3)
            # How fast must this train move (km/h) to make its deadline?
            # If required_speed > MAX_SPEED: impossible (agent must know this)
            dist_remaining_km = SECTION_LENGTH_KM * (1.0 - route_progress)
            if deadline_rem > 0:
                required_speed_kmh = dist_remaining_km / max(deadline_rem / 60.0, 0.01)
            else:
                required_speed_kmh = MAX_SPEED * 2.0   # already late — max urgency
            required_speed_norm = min(required_speed_kmh, MAX_SPEED * 2.0) / (MAX_SPEED * 2.0)

            # Assemble 24 features
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
            obs[i, 23] = required_speed_norm                 # required_speed (urgency)

            # Nearest available divert opportunity ahead (Phase 4 / GAP 1)
            nearest_loop_dist = self._get_nearest_loop_dist_km(train)
            obs[i, 24] = min(nearest_loop_dist, SECTION_LENGTH_KM) / SECTION_LENGTH_KM

        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        self.sim_time += 1
        reward      = 0.0
        terminated  = False
        term_reason = ""

        # Traffic tax — penalise congestion per active train
        num_active = sum(1 for t in self.trains
                         if not t['finished'] and t['position'] not in (0, 998))
        reward -= 0.005 * num_active

        # Build soft reservations map mapping platform node id -> reserving train id
        self._soft_reservations = {}
        for t in self.trains:
            if not t['finished'] and t.get('reserved_platform') is not None:
                r_plat = t['reserved_platform']
                if r_plat not in self._soft_reservations:
                    self._soft_reservations[r_plat] = []
                self._soft_reservations[r_plat].append(t['id'])

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

            # A station's entry switch (SWITCH type, more than one next
            # option) has no direct switch-in → switch-out bypass: every
            # train rides an explicit platform/loop track through the
            # station, whether or not it dwells there. So act==1
            # (PROCEED_MAIN) is treated identically to act==2 (DIVERT_LOOP)
            # at these nodes — both resolve via _select_divert_target — and
            # only degenerates to a bare next_opts[0] hop at ordinary,
            # non-branching nodes (plain mainline blocks, switch-out nodes,
            # mid-section crossing-loop approaches, etc).
            _entry_next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
            is_branching_entry = node_data.get('type') == 'SWITCH' and len(_entry_next_opts) > 1

            # Default "next" target for display purposes — the main/through track.
            # Overwritten below with the real committed target for MAIN/DIVERT moves.
            _display_next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
            
            # EARLY RESERVATION / PEEK (direct — train is at switch with fan-out)
            if _display_next_opts and len(_display_next_opts) > 1:
                _main_target = _display_next_opts[0]
                _loop_targets = [n for n in _display_next_opts if n != _main_target]
                _st_name = self.track_map.get(_loop_targets[0], {}).get('station') if _loop_targets else None
                
                if _st_name:
                    if train.get('reserved_platform') is None:
                        train['_newly_reserved'] = True
                        self._select_divert_target(train, _loop_targets, direction)
                        # Mark that the physical move toward this reservation
                        # must be deferred by one tick so the frontend sees
                        # reserved_platform set BEFORE the train's position changes.
                        if train.get('reserved_platform') is not None:
                            train['_divert_move_deferred'] = True

            # TWO-HOP EARLY RESERVATION PEEK — train is one node BEFORE the switch.
            # When current node has a single next hop that is itself a SWITCH with
            # multiple fan-out options (platform/loop targets), pre-reserve the platform
            # now so that the frontend switch-zone rendering tick (edge-{cur}-{switch})
            # already has a non-null reserved_platform to compute the correct visual row.
            # Without this, reserved_platform is only set AFTER the train physically
            # arrives at the switch, one broadcast tick too late — causing the lateral
            # track-jump bug on the approach edge.
            elif (len(_display_next_opts) == 1 and train.get('reserved_platform') is None):
                _single_next = _display_next_opts[0]
                _single_next_data = self.track_map.get(_single_next, {})
                if _single_next_data.get('type') == 'SWITCH':
                    _lookahead_opts = (
                        _single_next_data.get('prev', []) if direction == 'UP'
                        else _single_next_data.get('next', [])
                    )
                    if len(_lookahead_opts) > 1:
                        _lh_main = _lookahead_opts[0]
                        _lh_loops = [n for n in _lookahead_opts if n != _lh_main]
                        _lh_st_name = self.track_map.get(_lh_loops[0], {}).get('station') if _lh_loops else None
                        if _lh_st_name:
                            train['_newly_reserved'] = True
                            train['_early_reservation'] = True
                            self._select_divert_target(train, _lh_loops, direction)
                            # Same deferred-move flag — the two-hop peek fires one tick
                            # BEFORE the train reaches the switch, so the reservation
                            # exists before act==2 ever runs.  Without this flag, the
                            # execution block would see had_reservation=True and move
                            # immediately on the very first act==2 tick.
                            if train.get('reserved_platform') is not None:
                                train['_divert_move_deferred'] = True

            # committed_next_node is now set AFTER the act block (see below)

            # ── Banker attach/detach wait ─────────────────────────────────
            if train.get('banker_wait', 0) > 0:
                train['banker_wait'] -= 1
                if train['banker_wait'] == 0:
                    if train.get('banker_wait_action') == 'ATTACH':
                        train['banker_attached'] = True
                    elif train.get('banker_wait_action') == 'DETACH':
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

            # ── Braking override ──────────────────────────────────────────
            sig_val, sig_dist_km = self.check_signal(i)
            if train['speed'] > 0 and sig_val == 2.0:
                d_brake = (train['speed'] ** 2) / (2.0 * train['decel_rate'] * 60.0)
                if sig_dist_km <= d_brake:
                    train['target_speed'] = 0

            # ── MAIN (act == 1) — only meaningful at non-branching nodes ──
            if act == 1 and not is_branching_entry:
                next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
                if not next_opts:
                    # Nowhere to go (e.g., edge removed by TOTAL_BLOCK)
                    reward -= 0.05
                    current_positions.append(pos)
                    continue
                target_node = next_opts[0]
                cap = self.track_map.get(target_node, {}).get('capacity', 1)
                dir_cap = max(1, cap // 2) if cap > 1 else cap
                if self.get_node_occupancy(target_node, direction) >= dir_cap:
                    # Cannot enter - directional track is full
                    reward -= 0.05
                    current_positions.append(pos)
                    continue
                train['target_speed'] = min(track_limit, train['max_speed'])
            elif act == 2 or (act == 1 and is_branching_entry):
                # DIVERT_LOOP, or PROCEED_MAIN at a branching station-entry
                # switch (redirected — see is_branching_entry above).
                next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
                if not next_opts:
                    reward -= 0.05
                    current_positions.append(pos)
                    continue
                main_target = next_opts[0]
                loop_targets = [n for n in next_opts if n != main_target]

                # Remember what was reserved BEFORE this call.  Early-peek code
                # (lines above) may have already written reserved_platform 1-2
                # ticks ago, so we can't use "was it None before" as our gate.
                # Instead we compare by VALUE: if _select_divert_target picks a
                # DIFFERENT node than what was already stored, the reservation is
                # newly committed this tick and we must set the deferred-move flag.
                _reserved_before = train.get('reserved_platform')

                target_node = self._select_divert_target(train, loop_targets, direction)
                # Cache for the execution block — avoids a second call that would
                # re-mutate reserved_platform and double-count soft reservations.
                _divert_target_node = target_node

                # If a loop target was chosen AND the reservation is genuinely new
                # (different from what was stored before this call) AND the deferred-
                # move flag hasn't been set already by an early-peek, set it now.
                # This covers the case where the train reached the switch without
                # the peek code having fired (e.g. topology has no one-hop approach
                # node, or approach node lacked the SWITCH type marker).
                if (target_node is not None
                        and target_node != _reserved_before
                        and not train.get('_divert_move_deferred')):
                    train['_divert_move_deferred'] = True

                if target_node is None:
                    train.pop('_divert_move_deferred', None)  # clear stale flag
                    target_node = main_target
                    cap = self.track_map.get(target_node, {}).get('capacity', 1)
                    dir_cap = max(1, cap // 2) if cap > 1 else cap
                    if self.get_node_occupancy(target_node, direction) >= dir_cap:
                        reward -= 0.05
                        current_positions.append(pos)
                        continue

                train['target_speed'] = min(track_limit, train['max_speed'])
            else:
                train['target_speed'] = 0

            # A1: committed_next_node — computed AFTER the act block so that
            # reserved_platform reflects the current-tick decision. It must be a direct
            # next hop from current node to form a valid topological edge.
            if train.get('reserved_platform') is not None and train['reserved_platform'] in _display_next_opts:
                train['committed_next_node'] = train['reserved_platform']
            else:
                fallback_node = _display_next_opts[0] if _display_next_opts else pos
                train['committed_next_node'] = fallback_node

            # A2: awaiting_platform — True only when the train is at a switch
            # node that feeds a station it has a scheduled stop at, AND no
            # platform has been reserved yet this tick.  False for all other
            # cases (pass-throughs, trains not near a switch, post-reservation).
            _aw_platform = False
            if _display_next_opts and len(_display_next_opts) > 1:
                _aw_main  = _display_next_opts[0]
                _aw_loops = [n for n in _display_next_opts if n != _aw_main]
                _aw_st    = self.track_map.get(_aw_loops[0], {}).get('station') if _aw_loops else None
                if _aw_st and self._is_scheduled_stop(train, _aw_st) and train.get('reserved_platform') is None:
                    _aw_platform = True
            train['awaiting_platform'] = _aw_platform


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
            # We let this reach >= 1.0 to trigger the block advance below.
            # The UI in main.py already defensively clamps the visual read to 0.999
            # so we don't get snap-back.
            # Add speed * sim_speed_factor
            speed_factor = getattr(self, 'sim_speed_factor', 1.0)
            self._movement_acc[i] += (train['speed'] / 60.0) * speed_factor

            # ── HOLD ──────────────────────────────────────────────────────
            if act == 0:
                priority_ratio = train['priority'] / 6.0
                node_type = node_data.get('type', '')
                if 'MAIN' in node_type or 'GHAT' in node_type:
                    if sig_val == 0.0 and train['speed'] == 0:
                        reward -= 0.02 * priority_ratio  # idle on main line — bad
                else:
                    if train.get('dwell_rem', 0) == 0:
                        reward -= 0.01 * priority_ratio  # loitering in loop/switch after dwell done
                    else:
                        reward -= 0.0005

                # FIX 5: Graduated idle penalty — smooth gradient from idle=10 to deadlock.
                # Ramps from 0.0 at idle=10 up to 0.5 at idle=120, giving the PPO critic
                # a continuous signal to learn from rather than a cliff at deadlock.
                if train['speed'] == 0 and train.get('dwell_rem', 0) == 0 and train.get('banker_wait', 0) == 0:
                    train['idle_time'] += 1
                    idle = train['idle_time']
                    if idle > 10:
                        idle_penalty = min((idle - 10) / 220.0, 0.5) * priority_ratio  # Scale by priority
                        reward -= idle_penalty

                current_positions.append(pos)

            # ── PROCEED or DIVERT ─────────────────────────────────────────
            else:
                moved_this_step = False
                while True:
                    next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
                    if not next_opts:
                        if not moved_this_step:
                            reward -= 0.01
                        current_positions.append(pos)
                        break

                    main_target = next_opts[0]
                    loop_targets = [n for n in next_opts if n != main_target]

                    if (act == 2 or is_branching_entry) and loop_targets:
                        # Reuse the target already resolved in the speed-calc block;
                        # do NOT call _select_divert_target again (would re-mutate
                        # reserved_platform and double-count soft reservations).
                        target_node = _divert_target_node
                        if target_node is None:
                            target_node = main_target
                            if not moved_this_step:
                                reward -= 0.02
                        else:
                            # _divert_move_deferred is set the tick a NEW reservation
                            # is first committed.  Pop it (consume once) to skip the
                            # physical move this tick so the frontend has one broadcast
                            # to learn the new reserved_platform before the train jumps.
                            if train.pop('_divert_move_deferred', False):
                                current_positions.append(pos)
                                break

                            if train['priority'] < 5:
                                main_occ = self.get_node_occupancy(main_target)
                                main_cap = self.track_map.get(main_target, {}).get('capacity', 1)
                                tgt_type = self.track_map.get(target_node, {}).get('type', '')
                                threshold = 1 if tgt_type == 'CROSSING_LOOP' else max(1, main_cap // 2)
                                if main_occ >= threshold:
                                    delay_val = max(0, self.sim_time - sched.get('deadline', 99999))
                                    if delay_val == 0 and not moved_this_step:
                                        reward += 0.3

                            if train['priority'] < 5:
                                my_km = self._get_train_km(train)
                                kasara_km  = self.station_nodes.get('KASARA',   {}).get('km', 0)
                                igatpuri_km = self.station_nodes.get('IGATPURI', {}).get('km', 0)
                                ghat_approach_km = 15.0
                                at_ghat_approach = (
                                    (direction == 'UP'   and abs(my_km - kasara_km)   < ghat_approach_km) or
                                    (direction == 'DOWN' and abs(my_km - igatpuri_km) < ghat_approach_km)
                                )
                                tok = self.ghat_token.status()
                                opposing_in_ghat = (not tok['is_free'] and tok['direction'] != direction)
                                if at_ghat_approach and opposing_in_ghat:
                                    delay_val = max(0, self.sim_time - sched.get('deadline', 99999))
                                    if delay_val == 0 and not moved_this_step:
                                        reward += 0.8
                    else:
                        target_node = main_target

                    # (committed_next_node assignment moved down to commit phase to prevent UI oscillation)

                    # Determine physical distance to target block
                    dist_to_next = abs(self.get_node_km(target_node) - self.get_node_km(pos))
                    
                    # Force a 1.0km physical distance for intra-station movements so trains 
                    # don't instantly teleport across the 180px station graphic in one tick.
                    node_st = self.track_map.get(pos, {}).get('station')
                    tgt_st = self.track_map.get(target_node, {}).get('station')
                    if node_st and tgt_st and node_st == tgt_st:
                        dist_to_next = max(1.0, dist_to_next)
                    else:
                        dist_to_next = max(0.1, dist_to_next) # minimum threshold

                    if self._movement_acc[i] < dist_to_next:
                        current_positions.append(pos)
                        break

                    # Token block entry check
                    if self._is_in_token_block(target_node):
                        if not self.ghat_token.can_enter(train['id'], direction):
                            train['speed'] = 0
                            train['target_speed'] = 0
                            self._train_speeds[i] = 0
                            if not moved_this_step:
                                reward -= 0.5
                                train['idle_time'] += 1
                            current_positions.append(pos)
                            break
                    else:
                        next_of_target = self.track_map.get(target_node, {}).get(
                            'prev' if direction == 'UP' else 'next', [])
                        if next_of_target and self._is_in_token_block(next_of_target[0]):
                            if not self.ghat_token.can_enter(train['id'], direction):
                                train['speed'] = 0
                                train['target_speed'] = 0
                                self._train_speeds[i] = 0
                                if not moved_this_step:
                                    train['idle_time'] += 1
                                current_positions.append(pos)
                                break

                    # Capacity check at commit time
                    commit_cap_raw = self.track_map.get(target_node, {}).get('capacity', 1)
                    commit_cap = max(1, commit_cap_raw // 2) if commit_cap_raw > 1 else commit_cap_raw
                    commit_occ = self.get_node_occupancy(target_node, direction)
                    
                    look_ahead_ok = True
                    if self._is_chokepoint_node(target_node):
                        if not self._next_section_has_room(target_node, direction, directional_check=True, train_id=train['id'], actual_target=target_node):
                            look_ahead_ok = False
                            if train.get('speed', 0) > 0 or act != 0:
                                _log.debug(f"[LOOK-AHEAD DENIAL] Train {train['id']} ({direction}) denied entry into chokepoint Node {target_node} because the next section is full.")

                    if commit_occ >= commit_cap or not look_ahead_ok:
                        train['speed'] = 0
                        train['target_speed'] = 0
                        self._train_speeds[i] = 0
                        if not moved_this_step:
                            reward -= 0.1
                            train['idle_time'] += 1
                        current_positions.append(pos)
                        break

                    # Commit move
                    old_pos = pos
                    self._move_train(train, old_pos, target_node)
                    train['position'] = target_node
                    pos = target_node
                    node_data = self.track_map.get(pos, {}) # update node_data for next iteration

                    _commit_next_opts = node_data.get('prev', []) if direction == 'UP' else node_data.get('next', [])
                    _commit_is_branching_entry = node_data.get('type') == 'SWITCH' and len(_commit_next_opts) > 1
                    if (_commit_is_branching_entry
                            and train.get('reserved_platform') is not None
                            and train['reserved_platform'] in _commit_next_opts):
                        # Just arrived at a station's branching entry switch and the
                        # platform/loop is already reserved (usually via the early-peek
                        # reservation, well before physically reaching the switch) — show
                        # the real target immediately. Defaulting to _commit_next_opts[0]
                        # here (the switch_out bypass option, which sits at index 0 of
                        # every branching switch's next list) for even a single tick
                        # broadcasts the wrong edge to the frontend: the train badge
                        # visibly overshoots toward the far side of the station box, then
                        # snaps back to the correct platform the moment this corrects
                        # itself on the next tick.
                        train['committed_next_node'] = train['reserved_platform']
                    else:
                        fallback_node = _commit_next_opts[0] if _commit_next_opts else pos
                        train['committed_next_node'] = fallback_node
                    self._movement_acc[i] -= dist_to_next
                    if _commit_is_branching_entry:
                        # Prevent carry-over fractional km when entering a station switch zone.
                        # If p > 0 on the very first frame of a curve, the frontend interpolates
                        # from the old mainline Y straight to the new curve Y, causing a jarring
                        # diagonal zigzag. Forcing p=0.0 ensures a smooth glide from the very start.
                        self._movement_acc[i] = 0.0
                    moved_this_step = True
                    
                    # Clear reserved_platform only when we've physically left the station's track bundle
                    if train.get('reserved_platform'):
                        res_st = self.track_map.get(train['reserved_platform'], {}).get('station')
                        curr_st = self.track_map.get(pos, {}).get('station')
                        if res_st and curr_st != res_st:
                            train['reserved_platform'] = None
                        train.pop('_early_reservation', None)

                    # Token system update
                    was_in_token = self._is_in_token_block(old_pos)
                    now_in_token = self._is_in_token_block(target_node)

                    if now_in_token and not was_in_token:
                        print(f"[TOKEN] Train {train['id']} ENTERS token block at node {target_node}")
                        self.ghat_token.train_entered(train['id'], direction)
                    elif was_in_token and not now_in_token:
                        print(f"[TOKEN] Train {train['id']} EXITS token block at node {target_node} (from {old_pos})")
                        self.ghat_token.train_exited(train['id'])
                        if train.get('banker_required'):
                            train['banker_wait'] = BANKER_DETACH_TIME
                            train['banker_wait_action'] = 'DETACH'

                    target_station = self.track_map.get(target_node, {}).get('station')
                    if train.get('banker_required') and not train.get('banker_attached'):
                        if (direction == 'DOWN' and target_station == 'KASARA') or \
                           (direction == 'UP' and target_station == 'IGATPURI'):
                            if train.get('banker_wait', 0) == 0:
                                train['banker_wait'] = BANKER_ATTACH_TIME
                                train['banker_wait_action'] = 'ATTACH'

                    if (target_station and
                            self.track_map.get(target_node, {}).get('type') == 'PLATFORM' and
                            self._is_scheduled_stop(train, target_station)):
                        train['dwell_rem'] = DWELL_TIME_PLATFORM
                    elif (act == 2 or is_branching_entry) and self.track_map.get(target_node, {}).get('type') in ('LOOP', 'CROSSING_LOOP'):
                        train['dwell_rem'] = DWELL_TIME_LOOP

                    if target_node not in train['visited_nodes']:
                        train['visited_nodes'].add(target_node)
                        train['idle_time'] = 0
                        eff = train['speed'] / max(track_limit, 1)
                        reward += eff * (train['priority'] / 6.0) * 0.05
                        reward += 0.3 * (train['priority'] / 6.0)
                    else:
                        reward -= 0.005

                    dest_node = (999 if direction == 'DOWN'
                                 else self.station_nodes['CSMT']['switch_out'])
                    if target_node == dest_node or not self.track_map.get(target_node, {}).get('next'):
                        train['position']    = 999
                        train['finished']    = True
                        train['finish_step'] = self.sim_time
                        train['speed']       = 0
                        self._move_train(train, target_node, 999)
                        reward += 30.0 * (1.0 + train['priority'] / 6.0)
                        current_positions.append(999)
                        break

                    # Cap movement to one committed node-hop per step() call per train
                    # to prevent "sliding" and allow frontend to render intermediate track curves.
                    current_positions.append(pos)
                    break

            # ── Punctuality penalty ───────────────────────────────────────
            # FIX 1: Linear penalty with hard cap — removes the quadratic
            # 'death spiral' that made the agent prefer deadlock over lateness.
            # Max bleed is now 2.0 per step per train (predictable & bounded).
            delay = max(0, self.sim_time - sched['deadline'])
            if delay > 0:
                # Exponential multiplier: Base 1.5, centered around priority 6 (Express)
                # Goods (P2) gets ~0.2x penalty, Rajdhani (P10) gets ~5.0x penalty
                pri_multiplier = 1.5 ** (train['priority'] - 6)
                
                penalty_rate = delay * pri_multiplier * 0.02
                
                # Scale the cap as well, so high-priority trains can bleed significantly more
                # reward when delayed than low-priority trains.
                max_cap = 2.0 * pri_multiplier
                reward -= min(penalty_rate, max_cap)

        # ── Deadlock detection ────────────────────────────────────────────
        # FIX 9: Raised threshold 80 → 120 (Fix 5 graduated penalty handles
        # 10–120 range; hard kill only when truly stuck).
        # FIX 2: Projected penalty replaces flat -100.
        # Projects the cost of sitting frozen for all remaining steps so the
        # agent can never profit by choosing deadlock over lateness.
        # Threshold raised 120→300: only trains that are truly stuck (not at a signal/loop)
        # should trigger a deadlock. Token-block waits no longer count toward idle_time.
        if not getattr(self, 'inference_mode', False):
            deadlocked = [t for t in self.trains if t.get('idle_time', 0) > 300]
        else:
            deadlocked = []
        
        if deadlocked:
            print(f"\n{'!'*50}")
            print(f"🛑 DEADLOCK @ step {self.sim_time} — rescuing {[t['id'] for t in deadlocked]}")
            print(f"{'!'*50}\n")
            for t in deadlocked:
                # Heavy penalty for each rescued train
                reward -= 50.0
                # Teleport: advance to the next free node so the jam breaks
                direction = t.get('direction', 'DOWN')
                pos = t['position']
                next_opts = (self.track_map.get(pos, {}).get('prev', []) if direction == 'UP'
                             else self.track_map.get(pos, {}).get('next', []))
                rescued = False
                for nxt in next_opts:
                    if (self.get_node_occupancy(nxt) <
                            self.track_map.get(nxt, {}).get('capacity', 1)):
                        was_in_token = self._is_in_token_block(pos)
                        now_in_token = self._is_in_token_block(nxt)
                        self._move_train(t, pos, nxt)
                        t['position'] = nxt
                        rescued = True
                        if was_in_token and not now_in_token:
                            print(f"[TOKEN] Train {t['id']} EXITS token block via RESCUE at node {nxt} (from {pos})")
                            self.ghat_token.train_exited(t['id'])
                        break
                t['idle_time'] = 0   # reset so it doesn't re-trigger immediately
                if not rescued:
                    # Truly no escape — mark finished to unblock others
                    t['finished'] = True
                    reward -= 20.0
                    if self._is_in_token_block(pos):
                        print(f"[TOKEN] Train {t['id']} EXITS token block via DEATH at node {pos}")
                        self.ghat_token.train_exited(t['id'])


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
                # FIX 3: Collision is the absolute worst outcome — must exceed
                # max deadlock and max cumulative delay penalties to close the 
                # 'ram trains to exit' exploit route. 
                reward    -= 10000.0
                terminated = True
                term_reason = "Collision Detected"

        # ── Episode termination ───────────────────────────────────────────
        # All spawned trains finished
        spawned = [p for p in current_positions if p != 0]
        if spawned and all(p == 999 for p in spawned):
            reward    += 40.0
            terminated = True
            term_reason = "Success"

        # Timeout
        last_spawn      = max(s['start_time'] for s in self.schedule.values())
        max_allowed     = last_spawn + 1500
        if self.sim_time > max_allowed:
            _log.warning(f"[TIMEOUT] sim_time={self.sim_time} max={max_allowed}")
            reward -= 5000.0  # Massive penalty to prevent 'stall until timeout' exploit
            terminated = True
            term_reason = "Timeout Exceeded"

        # FIX 7: Normalise reward by √n_trains so reward magnitude stays
        # proportional across curriculum levels (L1→L6 would otherwise grow
        # 3–5× in absolute scale, destabilising the PPO advantage estimator).
        n_trains = max(len(self.trains), 1)
        reward = reward / (n_trains ** 0.5)

        info = {"termination_reason": term_reason} if terminated else {}
        return self._get_observation(), reward, terminated, False, info

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