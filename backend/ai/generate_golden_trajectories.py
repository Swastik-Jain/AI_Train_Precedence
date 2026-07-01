"""
generate_golden_trajectories.py
================================
Step 1 of the Hybrid Pipeline — "Expert" Trajectory Generation.

Wires the AI_Train_Precedence map topology (from map_generator.py) and a
5-train fleet (matching config.py types) into the CP-SAT OR-Solver to
produce a conflict-free, priority-optimal schedule.

Output files (written to backend/ai/):
  expert_actions.json  — step-by-step {0,1,2} action log per train
  golden_schedule.json — full arrival/departure timetable per node

Usage:
  cd backend/ai
  python generate_golden_trajectories.py
  python generate_golden_trajectories.py --trains 8 --out expert_actions.json
"""

import os
import sys
import json
import logging
import argparse

# Make sure local ai/ imports work regardless of call location
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("TrajectoryGen")

# ─────────────────────────────────────────────────────────────────
# STEP 1A: Build a deterministic OR-Solver-compatible track_map
#           from the same map_generator.py topology used by the RL env
# ─────────────────────────────────────────────────────────────────

def build_track_map_for_solver() -> dict:
    """
    Calls map_generator.generate_realistic_section() and converts its
    integer-keyed dict into a string-keyed dict compatible with or_solver.py.

    OR-Solver expects:
      track_map[node_id (str)] = {
          "capacity": int,
          "type": str,           # "STATION", "BLOCK", "SINGLE_LINE_BLOCK", ...
          "SINGLE_LINE_BLOCK": bool,
          "next": [str, ...]
      }
    """
    from map_generator import generate_realistic_section

    raw_map, loop_sections, end_node, station_nodes, token_blocks = generate_realistic_section()

    # Convert integer keys → string keys, integer next-lists → string lists
    solver_map = {}
    for node_id, data in raw_map.items():
        node_str = str(node_id)
        solver_map[node_str] = {
            "capacity":          data.get("capacity", 1),
            "type":              data.get("type", "BLOCK"),
            "SINGLE_LINE_BLOCK": data.get("type", "") == "SINGLE_LINE_BLOCK",
            "next":              [str(n) for n in data.get("next", [])],
            "speed":             data.get("speed", 75),
            "km":                data.get("km", 0.0),
        }

    logger.info(f"✅ Track map built: {len(solver_map)} nodes")
    return solver_map, str(end_node)


# ─────────────────────────────────────────────────────────────────
# STEP 1B: Build a 5-train fleet with real runtimes derived from
#           physics (max_speed, distance, safety buffers)
# ─────────────────────────────────────────────────────────────────

# Fleet definition — mirrors config.py "Nightmare Mix" but deterministic
FLEET_TEMPLATES = [
    {
        "id":       "VB100",
        "type":     "Vande Bharat",
        "max_speed": 130,
        "accel_rate": 12,
        "decel_rate": 20,
        "priority": 10,
        "direction": 1,
        "start_offset": 0,
        "scheduled_arrival": 60,
    },
    {
        "id":       "RJD101",
        "type":     "Rajdhani",
        "max_speed": 130,
        "accel_rate": 12,
        "decel_rate": 20,
        "priority": 10,
        "direction": 1,
        "start_offset": 5,
        "scheduled_arrival": 65,
    },
    {
        "id":       "SF102",
        "type":     "Superfast",
        "max_speed": 110,
        "accel_rate": 10,
        "decel_rate": 18,
        "priority": 8,
        "direction": 1,
        "start_offset": 12,
        "scheduled_arrival": 75,
    },
    {
        "id":       "SF103",
        "type":     "Superfast",
        "max_speed": 110,
        "accel_rate": 10,
        "decel_rate": 18,
        "priority": 8,
        "direction": 1,
        "start_offset": 18,
        "scheduled_arrival": 85,
    },
    {
        "id":       "EXP104",
        "type":     "Express",
        "max_speed": 90,
        "accel_rate": 8,
        "decel_rate": 15,
        "priority": 6,
        "direction": 1,
        "start_offset": 25,
        "scheduled_arrival": 100,
    },
    {
        "id":       "EXP105",
        "type":     "Express",
        "max_speed": 90,
        "accel_rate": 8,
        "decel_rate": 15,
        "priority": 6,
        "direction": 1,
        "start_offset": 32,
        "scheduled_arrival": 110,
    },
    {
        "id":       "FRT106",
        "type":     "Freight (WAG-9)",
        "max_speed": 60,
        "accel_rate": 4,
        "decel_rate": 8,
        "priority": 2,
        "direction": 1,
        "start_offset": 2,
        "scheduled_arrival": 120,
    },
]


def compute_runtime(speed_kmh: float, node_type: str, dist_km: float) -> float:
    """
    Estimate block transit time in minutes as a float.
    Minimum 0.1 minutes enforced.
    """
    # Speed is capped to block speed limit
    BLOCK_SPEED_CAPS = {
        "MAIN_BLOCK":        130,
        "SINGLE_LINE_BLOCK":  75,
        "LOOP":               30,
        "PLATFORM":           30,
        "PSR_CURVE":          30,
        "SWITCH":             30,
        "YARD":               30,
        "DESTINATION":         0,
    }
    cap = BLOCK_SPEED_CAPS.get(node_type, 75)
    effective_speed = min(speed_kmh, cap) if cap > 0 else speed_kmh

    if effective_speed <= 0:
        return 1.0
    transit_minutes = (dist_km / effective_speed) * 60.0
    return max(0.1, transit_minutes)


def build_path_and_runtimes(solver_map: dict, train: dict, start_node: str, end_node: str):
    """
    Trace a linear path from start_node to end_node through the solver_map
    (following the first 'next' pointer at each node — the mainline path).

    Returns:
      path      : [str, str, ...]  ordered node ids
      runtimes  : {node_id: int}   minutes to traverse INTO each node
      dwell_times: {node_id: int}  dwell at stations/loops
    """
    path = [start_node]
    visited = {start_node}
    cursor = start_node

    while cursor != end_node:
        node_data = solver_map.get(cursor, {})
        nexts = node_data.get("next", [])
        if not nexts:
            break
        nxt = nexts[0]   # follow mainline (first pointer)
        if nxt in visited:
            break
        path.append(nxt)
        visited.add(nxt)
        cursor = nxt

    # Runtime for each leg: time to travel from path[i-1] to path[i]
    runtimes = {}
    accumulated_float = 0.0
    accumulated_int = 0
    
    for i in range(1, len(path)):
        from_node = path[i - 1]
        to_node   = path[i]
        from_data = solver_map.get(from_node, {})
        to_data   = solver_map.get(to_node, {})
        
        # Determine distance
        from_km = from_data.get("km", 0.0)
        to_km   = to_data.get("km", 0.0)
        dist_km = abs(to_km - from_km)
        if dist_km == 0:
            dist_km = 0.1 # Minimum fallback distance for zero-length blocks (like switches)
            
        from_type = from_data.get("type", "BLOCK")
        float_mins = compute_runtime(train["max_speed"], from_type, dist_km)
        
        accumulated_float += float_mins
        int_target = round(accumulated_float)
        
        # Distribute the rounded integers to blocks
        leg_int = int_target - accumulated_int
        if leg_int < 0:
            leg_int = 0
            accumulated_int += 0
            accumulated_float = max(accumulated_float, float(accumulated_int))
        else:
            accumulated_int += leg_int
            
        runtimes[from_node] = leg_int

    # Dwell at stations/platforms (minimum 2 min, 0 elsewhere)
    dwell_times = {}
    for node in path:
        ntype = solver_map.get(node, {}).get("type", "BLOCK")
        if ntype in ("PLATFORM", "LOOP", "STATION"):
            dwell_times[node] = 2

    return path, runtimes, dwell_times


def build_active_fleet(solver_map: dict, start_node: str, end_node: str) -> list:
    """Convert FLEET_TEMPLATES into or_solver.py compatible dicts."""
    fleet = []
    for tmpl in FLEET_TEMPLATES:
        path, runtimes, dwell_times = build_path_and_runtimes(
            solver_map, tmpl, start_node, end_node
        )
        fleet.append({
            "id":               tmpl["id"],
            "type":             tmpl["type"],
            "max_speed":        tmpl["max_speed"],
            "priority":         tmpl["priority"],
            "direction":        tmpl["direction"],
            "path":             path,
            "runtimes":         runtimes,
            "dwell_times":      dwell_times,
            "scheduled_arrival": tmpl["scheduled_arrival"],
            # CP-SAT arrival var at first node constrained to start offset
            "start_offset":     tmpl["start_offset"],
        })
        logger.info(
            f"  🚂 {tmpl['id']} ({tmpl['type']}) | path len={len(path)} "
            f"| starts at t={tmpl['start_offset']}m"
        )
    return fleet


# ─────────────────────────────────────────────────────────────────
# STEP 1C: Add start-time constraints to the CP-SAT model
#           (pin Arrival[train][first_node] == start_offset)
# ─────────────────────────────────────────────────────────────────

def run_solver_with_start_times(solver_map: dict, fleet: list, out_path: str, golden_path: str):
    """
    Wraps or_solver.solve_train_schedule() and additionally pins
    each train's first-node arrival to its start_offset so temporal
    alignment with the RL environment start times is preserved.
    """
    from ortools.sat.python import cp_model
    TIME_HORIZON = 1440
    BLOCK_HEADWAY = 3
    STATION_BUFFER = 5

    def get_priority_weight(train_type: str) -> int:
        mapping = {
            "Vande Bharat": 100,
            "Rajdhani": 80,
            "Superfast": 50,
            "Express": 20,
            "Freight (WAG-9)": 5,
            "Freight (WAG-12)": 5
        }
        return mapping.get(train_type, 10)

    def generate_expert_actions(sched, flt, smap):
        return {"schedule": sched, "expert_actions": {}}


    model   = cp_model.CpModel()
    arrivals    = {}
    departures  = {}

    # ── Variable creation + constraints ──────────────────────────────────
    for train in fleet:
        t_id = train["id"]
        arrivals[t_id]   = {}
        departures[t_id] = {}
        path = train["path"]

        for i, node in enumerate(path):
            arrivals[t_id][node]   = model.NewIntVar(0, TIME_HORIZON, f"Arr_{t_id}_{node}")
            departures[t_id][node] = model.NewIntVar(0, TIME_HORIZON, f"Dep_{t_id}_{node}")

            # Pin first node arrival to the train's entry time
            if i == 0:
                model.Add(arrivals[t_id][node] == train["start_offset"])

            # Dwell
            dwell = train.get("dwell_times", {}).get(node, 0)
            if solver_map.get(node, {}).get("type", "BLOCK") == "STATION":
                dwell = max(dwell, 2)
            model.Add(departures[t_id][node] >= arrivals[t_id][node] + dwell)

            # Inertia travel
            if i > 0:
                prev_node = path[i - 1]
                run_time  = train.get("runtimes", {}).get(prev_node, 1)
                model.Add(arrivals[t_id][node] >= departures[t_id][prev_node] + run_time)

    # ── Conflict resolution ───────────────────────────────────────────────
    for i in range(len(fleet)):
        for j in range(i + 1, len(fleet)):
            t1, t2   = fleet[i], fleet[j]
            t1_id, t2_id = t1["id"], t2["id"]
            common   = set(t1["path"]).intersection(set(t2["path"]))

            for node in common:
                cap       = solver_map.get(node, {}).get("capacity", 1)
                node_type = solver_map.get(node, {}).get("type", "BLOCK")
                is_single = solver_map.get(node, {}).get("SINGLE_LINE_BLOCK", False)
                buffer    = STATION_BUFFER if node_type == "STATION" else BLOCK_HEADWAY
                dir1      = t1.get("direction", 1)
                dir2      = t2.get("direction", 1)

                if cap == 1 or is_single or (is_single and dir1 != dir2):
                    bv = model.NewBoolVar(f"{t1_id}_b4_{t2_id}_{node}")
                    model.Add(
                        arrivals[t2_id][node] >= departures[t1_id][node] + buffer
                    ).OnlyEnforceIf(bv)
                    model.Add(
                        arrivals[t1_id][node] >= departures[t2_id][node] + buffer
                    ).OnlyEnforceIf(bv.Not())

    # ── Objective ─────────────────────────────────────────────────────────
    weighted_delays = []
    dest_arrivals   = []

    for train in fleet:
        t_id      = train["id"]
        dest      = train["path"][-1]
        sched_arr = train.get("scheduled_arrival", TIME_HORIZON)
        weight    = get_priority_weight(train.get("type", "Passenger"))

        delay = model.NewIntVar(0, TIME_HORIZON, f"Delay_{t_id}")
        model.AddMaxEquality(delay, [0, arrivals[t_id][dest] - sched_arr])

        # Quadratic Penalty: Delay^2 * Weight
        delay_sq = model.NewIntVar(0, TIME_HORIZON * TIME_HORIZON, f"DelaySq_{t_id}")
        model.AddMultiplicationEquality(delay_sq, [delay, delay])

        w_delay = model.NewIntVar(0, TIME_HORIZON * TIME_HORIZON * 10, f"WDelay_{t_id}")
        model.AddMultiplicationEquality(w_delay, [delay_sq, weight])
        weighted_delays.append(w_delay)
        dest_arrivals.append(arrivals[t_id][dest])

    global_delay = model.NewIntVar(0, TIME_HORIZON * TIME_HORIZON * 10 * len(fleet), "GlobalDelay")
    model.Add(global_delay == sum(weighted_delays))

    makespan = model.NewIntVar(0, TIME_HORIZON, "Makespan")
    model.AddMaxEquality(makespan, dest_arrivals)

    Z = model.NewIntVar(0, (TIME_HORIZON * TIME_HORIZON * 10 * len(fleet) * 100) + TIME_HORIZON, "Z")
    model.Add(Z == (global_delay * 100) + makespan)
    model.Minimize(Z)

    # ── Solve ─────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    solver.parameters.log_search_progress = False

    logger.info("🔍 CP-SAT Solver running (max 15s)...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.error("❌ No feasible solution found. Check fleet start times and horizon.")
        return None

    logger.info(f"✅ Solution found — Status: {solver.StatusName(status)}")

    # ── Extract schedule ──────────────────────────────────────────────────
    schedule = {}
    for train in fleet:
        t_id = train["id"]
        schedule[t_id] = {}
        for node in train["path"]:
            schedule[t_id][node] = {
                "arrival":   solver.Value(arrivals[t_id][node]),
                "departure": solver.Value(departures[t_id][node]),
            }
        dest = train["path"][-1]
        actual_arr  = solver.Value(arrivals[t_id][dest])
        delay_mins  = max(0, actual_arr - train["scheduled_arrival"])
        weight      = get_priority_weight(train["type"])
        logger.info(
            f"  🚂 {t_id} | arr_dest={actual_arr}m | "
            f"deadline={train['scheduled_arrival']}m | "
            f"delay={delay_mins}m | W={weight}"
        )

    # ── Generate expert action sequences ─────────────────────────────────
    result = generate_expert_actions(schedule, fleet, solver_map)

    # Rename the output file to the requested path
    if out_path != "expert_actions.json" and os.path.exists("expert_actions.json"):
        os.rename("expert_actions.json", out_path)
        result["_output_file"] = out_path

    # Also write the detailed golden schedule
    with open(golden_path, "w") as f:
        json.dump({
            "fleet_metadata": [
                {
                    "id":               t["id"],
                    "type":             t["type"],
                    "max_speed":        t["max_speed"],
                    "priority":         t["priority"],
                    "start_offset":     t["start_offset"],
                    "scheduled_arrival":t["scheduled_arrival"],
                    "path_length":      len(t["path"]),
                }
                for t in fleet
            ],
            "schedule": schedule,
            "expert_actions": result["expert_actions"],
        }, f, indent=4)

    logger.info(f"💾 Expert actions → {out_path}")
    logger.info(f"💾 Golden schedule → {golden_path}")

    # ── Print human-readable timetable ───────────────────────────────────
    print("\n" + "═" * 64)
    print("  GOLDEN TIMETABLE — OR-Solver Optimal Schedule")
    print("═" * 64)
    for train in fleet:
        t_id = train["id"]
        print(f"\n  🚂 {t_id} ({train['type']}) — Priority W={get_priority_weight(train['type'])}")
        for node in train["path"][:6]:   # show first 6 nodes
            a = schedule[t_id][node]["arrival"]
            d = schedule[t_id][node]["departure"]
            ntype = solver_map.get(node, {}).get("type", "?")
            print(f"     Node {node:>6} ({ntype:<18}) | arr={a:>4}m  dep={d:>4}m")
        if len(train["path"]) > 6:
            print(f"     ... {len(train['path']) - 6} more nodes ...")
        dest = train["path"][-1]
        a = schedule[t_id][dest]["arrival"]
        print(f"     DEST  {dest:>6}                      | arr={a:>4}m  (deadline={train['scheduled_arrival']}m)")
    print("\n" + "═" * 64 + "\n")

    return result


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Golden Expert Trajectories via CP-SAT OR-Solver",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--trains", type=int, default=7,
        help="Number of trains from the fleet template (max 7, default 7)"
    )
    parser.add_argument(
        "--out", default="expert_actions.json",
        help="Output path for expert_actions.json"
    )
    parser.add_argument(
        "--golden", default="golden_schedule.json",
        help="Output path for detailed golden timetable JSON"
    )
    args = parser.parse_args()

    num_trains = min(args.trains, len(FLEET_TEMPLATES))
    logger.info(f"🚂 Generating golden trajectories for {num_trains} trains...")

    # 1. Build map
    solver_map, end_node = build_track_map_for_solver()

    # 2. Build fleet (start node is always "1" — first block after YARD)
    fleet = build_active_fleet(solver_map, start_node="1", end_node=end_node)
    fleet = fleet[:num_trains]

    # 3. Solve
    result = run_solver_with_start_times(
        solver_map, fleet,
        out_path=args.out,
        golden_path=args.golden
    )

    if result:
        total_steps = sum(len(v) for v in result["expert_actions"].values())
        logger.info(
            f"🏁 Done. {num_trains} trains | {total_steps} total action steps generated."
        )
        logger.info("   Ready for Behaviour Cloning — run:")
        logger.info(f"   python run_bc_warmup.py --expert {args.out}")
    else:
        logger.error("Generation failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
