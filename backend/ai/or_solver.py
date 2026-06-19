def solve_train_schedule(track_map, active_fleet):
    """
    Mock OR-Tools solver.
    Returns a simple greedy schedule so the backend doesn't crash if the real solver is missing.
    """
    schedule = {}
    for train in active_fleet:
        t_id = train["id"]
        path = train["path"]
        sched = []
        t_time = 0
        for node in path:
            sched.append({
                "node": node,
                "arrival": t_time,
                "departure": t_time + 2
            })
            t_time += 10
        schedule[t_id] = sched

    return {
        "schedule": schedule,
        "expert_actions": {}
    }
