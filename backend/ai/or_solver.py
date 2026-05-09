import json
import logging
from ortools.sat.python import cp_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OR-Solver")

# --- 1. Global Ground Truths & Constants ---
TIME_HORIZON = 120  # Minutes
BLOCK_HEADWAY = 3   # Minutes
STATION_BUFFER = 5  # Minutes

PRIORITY_WEIGHTS = {
    "Vande Bharat": 10,
    "Rajdhani": 10,
    "Superfast": 8,
    "Express": 6,
    "Local": 5,
    "Suburban": 5,
    "Passenger": 3,
    "Freight (WAG-9)": 2,
    "Maintenance": 1,
    "Special": 1
}

def get_priority_weight(train_type: str) -> int:
    return PRIORITY_WEIGHTS.get(train_type, 3)  # default to passenger priority


def solve_train_schedule(track_map: dict, active_fleet: list):
    """
    track_map: dict mapping node_id to node properties. 
               e.g. {"node_1": {"capacity": 2, "is_single_line": False, "type": "STATION"}}
    active_fleet: list of train dictionary configs.
               e.g. {"id": "T1", "type": "Rajdhani", "path": ["node_1", "node_2"], 
                     "scheduled_arrival": 60, "runtimes": {"node_1": 5}, "dwell_times": {"node_1": 2}, "direction": 1}
    """
    model = cp_model.CpModel()
    
    arrivals = {}
    departures = {}
    
    # Initialize variables for all N trains
    for train in active_fleet:
        t_id = train["id"]
        arrivals[t_id] = {}
        departures[t_id] = {}
        path = train["path"]
        
        for i, node in enumerate(path):
            arrivals[t_id][node] = model.NewIntVar(0, TIME_HORIZON, f"Arr_{t_id}_{node}")
            departures[t_id][node] = model.NewIntVar(0, TIME_HORIZON, f"Dep_{t_id}_{node}")
            
            # --- 2. Mathematical Constraints ---
            
            # Station Operations / Minimum Dwell Time
            dwell = train.get("dwell_times", {}).get(node, 0)
            if track_map.get(node, {}).get("type", "BLOCK") == "STATION":
                dwell = max(dwell, 2)  # Platform dwell >= 2 minutes
                
            model.Add(departures[t_id][node] >= arrivals[t_id][node] + dwell)
            
            # Inertia-Aware Travel
            if i > 0:
                prev_node = path[i-1]
                run_time = train.get("runtimes", {}).get(prev_node, 1)
                model.Add(arrivals[t_id][node] >= departures[t_id][prev_node] + run_time)
                
    # --- Conflict Resolution (The Safety Shield) ---
    for i in range(len(active_fleet)):
        for j in range(i + 1, len(active_fleet)):
            t1 = active_fleet[i]
            t2 = active_fleet[j]
            t1_id = t1["id"]
            t2_id = t2["id"]
            
            common_nodes = set(t1["path"]).intersection(set(t2["path"]))
            
            for node in common_nodes:
                cap = track_map.get(node, {}).get("capacity", 1)
                node_type = track_map.get(node, {}).get("type", "BLOCK")
                is_single = track_map.get(node, {}).get("SINGLE_LINE_BLOCK", False)
                
                buffer = STATION_BUFFER if node_type == "STATION" else BLOCK_HEADWAY
                
                # Single-Track Crossing Logic / Capacity logic
                # For any two trains competing for a block with restricted capacity 
                # OR opposite direction trains on single line blocks (they must only meet at cap > 1)
                # UPDATED: For 7-train density, strictly enforce SINGLE_LINE_BLOCK occupancy (max 1).
                dir1 = t1.get("direction", 1)
                dir2 = t2.get("direction", 2)
                
                if cap == 1 or is_single or (is_single and dir1 != dir2):
                    # Enforce strict ordering
                    is_t1_before_t2 = model.NewBoolVar(f"{t1_id}_before_{t2_id}_{node}")
                    
                    # If is_t1_before_t2 is True: Arrival(t2) >= Departure(t1) + buffer
                    model.Add(arrivals[t2_id][node] >= departures[t1_id][node] + buffer).OnlyEnforceIf(is_t1_before_t2)
                    
                    # If is_t1_before_t2 is False: Arrival(t1) >= Departure(t2) + buffer
                    model.Add(arrivals[t1_id][node] >= departures[t2_id][node] + buffer).OnlyEnforceIf(is_t1_before_t2.Not())

    # --- 3. Optimization Objective ---
    # Z = Sum(W_i * max(0, Arrival_{i, dest} - ScheduledArrival_i))
    weighted_delays = []
    dest_arrivals = []
    
    for train in active_fleet:
        t_id = train["id"]
        if not train["path"]:
            continue
            
        dest = train["path"][-1]
        sched_arr = train.get("scheduled_arrival", TIME_HORIZON)
        weight = get_priority_weight(train.get("type", "Passenger"))
        
        # Delay variable
        delay = model.NewIntVar(0, TIME_HORIZON, f"Delay_{t_id}")
        model.AddMaxEquality(delay, [0, arrivals[t_id][dest] - sched_arr])
        
        # QUADRATIC PENALTY: Penalty = Priority * Delay^2
        delay_sq = model.NewIntVar(0, TIME_HORIZON * TIME_HORIZON, f"DelaySq_{t_id}")
        model.AddMultiplicationEquality(delay_sq, [delay, delay])
        
        weighted_delay = model.NewIntVar(0, TIME_HORIZON * TIME_HORIZON * 10, f"WeightedDelay_{t_id}")
        model.AddMultiplicationEquality(weighted_delay, [delay_sq, weight])
        weighted_delays.append(weighted_delay)
        
        dest_arrivals.append(arrivals[t_id][dest])
        
    global_delay = model.NewIntVar(0, TIME_HORIZON * TIME_HORIZON * 10 * len(active_fleet), "GlobalWeightedDelay")
    model.Add(global_delay == sum(weighted_delays))
    
    # Secondary objective: Maximize throughput by minimizing the total time the last train exits
    makespan = model.NewIntVar(0, TIME_HORIZON, "Makespan")
    if dest_arrivals:
        model.AddMaxEquality(makespan, dest_arrivals)
        
    # Scale Z to heavily favor the primary objective
    # Note: Global delay is now quadratic, so we use a larger factor for Z
    Z = model.NewIntVar(0, (TIME_HORIZON * TIME_HORIZON * 10 * len(active_fleet) * 100) + TIME_HORIZON, "Z")
    model.Add(Z == (global_delay * 100) + makespan)
    
    model.Minimize(Z)
    
    # --- 4. System Integration & Output ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    
    logger.info("Starting SAT Solver...")
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logger.info(f"Solution found. Status: {solver.StatusName(status)}")
        
        schedule = {}
        for train in active_fleet:
            t_id = train["id"]
            schedule[t_id] = {}
            for node in train["path"]:
                schedule[t_id][node] = {
                    "arrival": solver.Value(arrivals[t_id][node]),
                    "departure": solver.Value(departures[t_id][node])
                }
        
        return generate_expert_actions(schedule, active_fleet, track_map)
    else:
        logger.warning(f"No feasible solution found up to {TIME_HORIZON} minutes limit.")
        return None

def generate_expert_actions(schedule, active_fleet, track_map):
    """
    Generates step-by-step action log:
      0: STOP (Wait in loop/at signal)
      1: MAIN (Proceed on main line)
      2: DIVERT (Enter loop/platform)
    """
    expert_actions = {}
    
    for train in active_fleet:
        t_id = train["id"]
        actions_by_minute = []
        path = train["path"]
        
        if not path:
            continue
            
        t_sched = schedule[t_id]
        
        start_time = t_sched[path[0]]["arrival"]
        end_time = t_sched[path[-1]]["departure"]
        
        current_node_idx = 0
        
        # Populate minute by minute
        for minute in range(end_time + 1):
            if minute < start_time:
                # Haven't started yet
                actions_by_minute.append(0) # STOP
                continue
                
            node = path[current_node_idx]
            n_arr = t_sched[node]["arrival"]
            n_dep = t_sched[node]["departure"]
            
            # If train is currently dwelling at a node
            if n_arr <= minute < n_dep:
                # If node is a station, assume we divert to a platform loop. Else we are just stopped.
                is_station = track_map.get(node, {}).get("type", "BLOCK") == "STATION"
                if is_station:
                    actions_by_minute.append(2)  # DIVERT to platform/loop
                else:
                    actions_by_minute.append(0)  # STOP / Wait at signal
            
            # If train has departed and is en-route to next node
            elif minute >= n_dep:
                if current_node_idx < len(path) - 1:
                    next_node = path[current_node_idx + 1]
                    next_n_arr = t_sched[next_node]["arrival"]
                    
                    if minute < next_n_arr:
                        actions_by_minute.append(1) # MAIN - moving
                    else:
                        # Reached next node
                        current_node_idx += 1
                        minute_val = 1 # recalculate via loop logic naturally in next iteration normally, but for safety: 
                        # To keep it exact, we will just say it starts arriving.
                        actions_by_minute.append(1) 
                else:
                    # Reached final destination
                    actions_by_minute.append(0)
                        
        expert_actions[t_id] = actions_by_minute
        
    # Write to File
    output_payload = {
        "schedule": schedule,
        "expert_actions": expert_actions
    }
    
    with open("expert_actions.json", "w") as f:
        json.dump(output_payload, f, indent=4)
        
    logger.info("Expert actions successfully written to expert_actions.json")
    return output_payload

if __name__ == "__main__":
    import os
    
    # Demo Mock Input Load
    if os.path.exists("input_data.json"):
        with open("input_data.json", "r") as f:
            data = json.load(f)
            solve_train_schedule(data.get("track_map", {}), data.get("active_fleet", []))
    else:
        logger.info("No input_data.json found when running directly. Import and pass inputs to solve_train_schedule.")
