"""
corridor_planner.py — Horizon CP-SAT Corridor Planner
Macro-level offline planner for CSMT-Manmad corridor.

Replaces or_solver.py, rewritten for:
    - Simulation steps (not minutes)
    - New train fleet structure (direction, stops, banker_required)
    - Token block constraints (Kasara-Igatpuri ghat mid-line)
    - Integer priority weights matching config.py (0-6 scale)
    - No pre-defined path required — uses station stops list

Primary use: Generate behavioral cloning (expert action) data
to warm-start RL training at higher curriculum levels.

Secondary use: Standalone CP-SAT baseline for benchmark comparison.

Usage:
    planner = CorridorPlanner(track_map, station_nodes, token_blocks)
    result  = planner.solve(trains, schedule, sim_time=0)
    # result['expert_actions'][train_id] = list of (step, action) tuples
    # result['schedule']                 = optimized arrival/departure steps
"""

import logging
import numpy as np
from collections import defaultdict
from ortools.sat.python import cp_model

from ai.config import SECTION_LENGTH_KM
from ai.map_generator import STATIONS, STATION_ORDER

_log = logging.getLogger("CorridorPlanner")

# Planning horizon in simulation steps
PLAN_HORIZON = 600      # ~10 hours of sim time

# Minimum headway between trains at same station (steps)
MIN_HEADWAY_STEPS = 3

# Token block exclusive use window (steps)
TOKEN_BLOCK_WINDOW = 8  # steps to traverse full ghat (3 blocks)

# Solver timeout
SOLVER_TIMEOUT = 10.0   # seconds — offline planning, can be generous

# Priority scale — maps integer priority to CP-SAT weight
PRIORITY_WEIGHTS = {
    6: 10,   # Rajdhani
    5: 8,    # Superfast
    4: 6,    # Mail/Express
    3: 4,    # Passenger
    2: 3,    # (unused currently)
    1: 2,    # Goods/Freight
    0: 1,    # Banker
}


class CorridorPlanner:
    """
    CP-SAT horizon planner for CSMT-Manmad corridor.

    Builds a constraint model over PLAN_HORIZON steps.
    Outputs optimized arrival/departure steps per station per train,
    and a step-by-step expert action sequence.
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

        # Station km lookup
        self._station_km = {
            name: data['km']
            for name, data in station_nodes.items()
        }

        # Precompute inter-station travel steps
        # Based on distance and typical speed for each segment
        self._travel_steps = self._precompute_travel_steps()

        _log.info(
            f"CorridorPlanner initialized | "
            f"stations={list(station_nodes.keys())} | "
            f"token_blocks={sorted(token_blocks)}"
        )

    def solve(
        self,
        trains: list,
        schedule: dict,
        sim_time: int = 0,
    ) -> dict:
        """
        Solve the corridor scheduling problem.

        Parameters
        ----------
        trains   : list of train dicts from env.trains
        schedule : env.schedule dict
        sim_time : current simulation timestep (planning starts here)

        Returns
        -------
        dict with keys:
            'status'         : 'OPTIMAL' | 'FEASIBLE' | 'INFEASIBLE'
            'schedule'       : {train_id: {station: {arrival, departure}}}
            'expert_actions' : {train_id: [(step, action), ...]}
            'total_delay'    : weighted delay score
        """
        model  = cp_model.CpModel()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = SOLVER_TIMEOUT

        horizon = PLAN_HORIZON

        # Filter active trains
        active = [
            t for t in trains
            if not t['finished'] and t['position'] not in (999,)
        ]

        if not active:
            return {'status': 'INFEASIBLE', 'schedule': {}, 'expert_actions': {}}

        # ── Variables ─────────────────────────────────────────────────────
        arrivals   = {}   # arrivals[train_id][station] = IntVar
        departures = {}   # departures[train_id][station] = IntVar

        for train in active:
            tid       = train['id']
            direction = train['direction']
            sched     = schedule.get(tid, {})
            stops     = sched.get('stops', [])
            start     = sched.get('start_time', sim_time)

            arrivals[tid]   = {}
            departures[tid] = {}

            # Determine ordered stops based on direction
            if direction == 'DOWN':
                ordered_stops = [
                    s for s in STATION_ORDER
                    if s in stops and s in self.station_nodes
                ]
            else:
                ordered_stops = [
                    s for s in reversed(STATION_ORDER)
                    if s in stops and s in self.station_nodes
                ]

            if not ordered_stops:
                continue

            # Add banker stops if required
            if train.get('banker_required'):
                banker_stops = (
                    ['KASARA', 'IGATPURI'] if direction == 'DOWN'
                    else ['IGATPURI', 'KASARA']
                )
                # Insert banker stops if not already in stops
                for bs in banker_stops:
                    if bs not in ordered_stops and bs in self.station_nodes:
                        # Find insertion point
                        bs_km = self._station_km.get(bs, 0)
                        inserted = False
                        for idx, stop in enumerate(ordered_stops):
                            stop_km = self._station_km.get(stop, 0)
                            if direction == 'DOWN' and stop_km > bs_km:
                                ordered_stops.insert(idx, bs)
                                inserted = True
                                break
                            elif direction == 'UP' and stop_km < bs_km:
                                ordered_stops.insert(idx, bs)
                                inserted = True
                                break
                        if not inserted:
                            ordered_stops.append(bs)

            train['_ordered_stops'] = ordered_stops

            prev_dep = None
            for idx, station in enumerate(ordered_stops):
                arr_var = model.NewIntVar(
                    start, horizon, f'arr_{tid}_{station}'
                )
                dep_var = model.NewIntVar(
                    start, horizon, f'dep_{tid}_{station}'
                )

                arrivals[tid][station]   = arr_var
                departures[tid][station] = dep_var

                # Dwell constraint
                dwell = self._get_dwell(train, station)
                model.Add(dep_var >= arr_var + dwell)

                # Travel time from previous stop
                if prev_dep is not None and idx > 0:
                    prev_station  = ordered_stops[idx - 1]
                    travel        = self._get_travel_steps(
                        prev_station, station, train
                    )
                    model.Add(arr_var >= prev_dep + travel)

                # Banker attachment delay at Kasara (UP) or Igatpuri (DOWN)
                if (station in ('KASARA', 'IGATPURI')
                        and train.get('banker_required')
                        and not train.get('banker_required') == False):
                    from ai.config import BANKER_ATTACH_TIME
                    model.Add(dep_var >= arr_var + BANKER_ATTACH_TIME)

                prev_dep = dep_var

        # ── Conflict constraints ───────────────────────────────────────────

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                t1  = active[i]
                t2  = active[j]
                id1 = t1['id']
                id2 = t2['id']

                stops1 = set(t1.get('_ordered_stops', []))
                stops2 = set(t2.get('_ordered_stops', []))
                common = stops1 & stops2

                for station in common:
                    if (id1 not in arrivals or station not in arrivals[id1]
                            or id2 not in arrivals or station not in arrivals[id2]):
                        continue

                    # Platform capacity at this station
                    sdata = self.station_nodes.get(station, {})
                    n_platforms = len(sdata.get('platforms', [1]))
                    capacity = max(n_platforms, 1)

                    if capacity == 1:
                        # Strict ordering
                        t1_first = model.NewBoolVar(f'{id1}_b4_{id2}_{station}')
                        model.Add(
                            arrivals[id2][station] >=
                            departures[id1][station] + MIN_HEADWAY_STEPS
                        ).OnlyEnforceIf(t1_first)
                        model.Add(
                            arrivals[id1][station] >=
                            departures[id2][station] + MIN_HEADWAY_STEPS
                        ).OnlyEnforceIf(t1_first.Not())
                    else:
                        # Softer headway
                        t1_first = model.NewBoolVar(f'{id1}_b4_{id2}_{station}')
                        model.Add(
                            arrivals[id2][station] >=
                            arrivals[id1][station] + MIN_HEADWAY_STEPS
                        ).OnlyEnforceIf(t1_first)
                        model.Add(
                            arrivals[id1][station] >=
                            arrivals[id2][station] + MIN_HEADWAY_STEPS
                        ).OnlyEnforceIf(t1_first.Not())

                    # Token block (ghat) — opposing directions cannot overlap
                    if (station in ('KASARA', 'IGATPURI')
                            and t1.get('direction') != t2.get('direction')):
                        ghat_t1_first = model.NewBoolVar(
                            f'ghat_{id1}_b4_{id2}'
                        )
                        model.Add(
                            arrivals[id2][station] >=
                            departures[id1][station] + TOKEN_BLOCK_WINDOW
                        ).OnlyEnforceIf(ghat_t1_first)
                        model.Add(
                            arrivals[id1][station] >=
                            departures[id2][station] + TOKEN_BLOCK_WINDOW
                        ).OnlyEnforceIf(ghat_t1_first.Not())

        # ── Objective ─────────────────────────────────────────────────────
        weighted_delays = []

        for train in active:
            tid      = train['id']
            sched    = schedule.get(tid, {})
            deadline = sched.get('deadline', horizon)
            priority = train.get('priority', 1)
            weight   = PRIORITY_WEIGHTS.get(priority, 2)

            stops = train.get('_ordered_stops', [])
            if not stops or tid not in arrivals:
                continue

            dest = stops[-1]
            if dest not in arrivals.get(tid, {}):
                continue

            delay = model.NewIntVar(0, horizon, f'delay_{tid}')
            model.AddMaxEquality(
                delay,
                [model.NewConstant(0),
                 arrivals[tid][dest] - model.NewConstant(deadline)]
            )

            # Quadratic penalty: delay² × priority_weight
            delay_sq = model.NewIntVar(0, horizon * horizon, f'delay_sq_{tid}')
            model.AddMultiplicationEquality(delay_sq, [delay, delay])

            weighted = model.NewIntVar(
                0, horizon * horizon * 10, f'weighted_{tid}'
            )
            model.AddMultiplicationEquality(
                weighted,
                [delay_sq, model.NewConstant(weight)]
            )
            weighted_delays.append(weighted)

        if weighted_delays:
            total_delay = model.NewIntVar(
                0,
                horizon * horizon * 10 * len(active),
                'total_delay'
            )
            model.Add(total_delay == sum(weighted_delays))
            model.Minimize(total_delay)

        # ── Solve ─────────────────────────────────────────────────────────
        _log.info(f"Solving for {len(active)} trains over {horizon} steps...")
        status = solver.Solve(model)

        status_name = solver.StatusName(status)
        _log.info(f"Solver status: {status_name}")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {
                'status':         'INFEASIBLE',
                'schedule':       {},
                'expert_actions': {},
                'total_delay':    -1,
            }

        # ── Decode solution ────────────────────────────────────────────────
        decoded_schedule = {}
        for train in active:
            tid   = train['id']
            stops = train.get('_ordered_stops', [])
            if tid not in arrivals:
                continue

            decoded_schedule[tid] = {}
            for station in stops:
                if station in arrivals.get(tid, {}):
                    decoded_schedule[tid][station] = {
                        'arrival':   solver.Value(arrivals[tid][station]),
                        'departure': solver.Value(departures[tid][station]),
                    }

        expert_actions = self._generate_expert_actions(
            decoded_schedule, active, sim_time, horizon
        )

        total_delay_val = 0
        if weighted_delays:
            try:
                total_delay_val = solver.ObjectiveValue()
            except Exception:
                pass

        return {
            'status':         status_name,
            'schedule':       decoded_schedule,
            'expert_actions': expert_actions,
            'total_delay':    total_delay_val,
        }

    def _generate_expert_actions(
        self,
        schedule: dict,
        active: list,
        sim_time: int,
        horizon: int,
    ) -> dict:
        """
        Convert optimized schedule into step-by-step expert actions.

        Actions:
            0 = HOLD  (wait, dwell, banker attach)
            1 = PROCEED_MAIN
            2 = DIVERT (enter platform/loop at station)
        """
        expert_actions = {}

        for train in active:
            tid       = train['id']
            direction = train['direction']
            stops     = train.get('_ordered_stops', [])

            if tid not in schedule or not stops:
                continue

            t_sched = schedule[tid]
            actions = []   # list of (step, action) tuples

            for idx, station in enumerate(stops):
                if station not in t_sched:
                    continue

                arr  = t_sched[station]['arrival']
                dep  = t_sched[station]['departure']

                # Travel to this station: PROCEED_MAIN
                if idx > 0:
                    prev_station = stops[idx - 1]
                    if prev_station in t_sched:
                        prev_dep = t_sched[prev_station]['departure']
                        for step in range(prev_dep, arr):
                            actions.append((step, 1))   # PROCEED_MAIN

                # Dwell at station: DIVERT (enter platform)
                for step in range(arr, dep):
                    actions.append((step, 2))   # DIVERT to platform

            # After last station: PROCEED to destination
            if stops:
                last_station = stops[-1]
                if last_station in t_sched:
                    last_dep = t_sched[last_station]['departure']
                    for step in range(last_dep, min(last_dep + 50, horizon)):
                        actions.append((step, 1))   # PROCEED_MAIN to terminus

            expert_actions[tid] = actions

        return expert_actions

    def _get_dwell(self, train: dict, station: str) -> int:
        """Dwell steps at a station for this train."""
        from ai.config import DWELL_TIME_PLATFORM, BANKER_ATTACH_TIME

        sched = {}
        dwell = 0

        # Scheduled stop
        stops = sched.get('stops', train.get('_ordered_stops', []))
        if station in stops:
            dwell = max(dwell, DWELL_TIME_PLATFORM)

        # Banker attachment at ghat endpoints
        if (station in ('KASARA', 'IGATPURI')
                and train.get('banker_required')):
            dwell = max(dwell, BANKER_ATTACH_TIME)

        return max(dwell, 1)   # minimum 1 step

    def _get_travel_steps(
        self,
        from_station: str,
        to_station: str,
        train: dict,
    ) -> int:
        """
        Estimated travel steps between two adjacent stations for this train.
        Based on km distance and train max_speed.
        """
        cached = self._travel_steps.get((from_station, to_station))
        if cached:
            speed_factor = train['max_speed'] / 130.0   # normalize to Rajdhani speed
            return max(1, int(cached / max(speed_factor, 0.3)))
        return 10   # fallback

    def _precompute_travel_steps(self) -> dict:
        """
        Precompute base travel steps between adjacent stations
        at reference speed (Rajdhani: 130kmph).
        Steps = km_distance / (130 km/h / 60 steps/h)
        """
        travel = {}
        station_list = list(self.station_nodes.keys())

        for i in range(len(station_list)):
            for j in range(i + 1, len(station_list)):
                s1  = station_list[i]
                s2  = station_list[j]
                km1 = self._station_km.get(s1, 0)
                km2 = self._station_km.get(s2, 0)
                km  = abs(km2 - km1)
                # At 130kmph: 1 sim-step = 130/60 ≈ 2.17km
                steps = max(1, int(km / (130 / 60)))
                travel[(s1, s2)] = steps
                travel[(s2, s1)] = steps

        return travel
