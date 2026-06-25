from main import TRAIN_STATES, FLEET_REGISTRY

DELAY_THRESHOLD = 10.0
for t_id, state in TRAIN_STATES.items():
    if state.get('status') == 'Scheduled': continue
    reg = FLEET_REGISTRY.get(t_id, {})
    deadline = reg.get('deadline', 120)
    start_time = reg.get('start_time', 0)
    
    path = state.get('path', [])
    curr_edge = state.get('edge_id')
    path_len = len(path) if path else 1
    curr_idx = path.index(curr_edge) if curr_edge in path else 0
    edge_pct = state.get('position_percentage', 0.0)
    completion = (curr_idx + edge_pct) / path_len
    
    train_sim_time = state.get('sim_time', 0)
    travel_time_elapsed = train_sim_time
    total_travel_budget = max(1, deadline - start_time)
    
    expected_completion = min(1.0, travel_time_elapsed / total_travel_budget) if total_travel_budget > 0 else 1.0
    
    if completion < expected_completion:
        delay = (expected_completion - completion) * total_travel_budget
    else:
        delay = 0.0
        
    print(f'Train: {t_id}')
    print(f'  deadline: {deadline}, start_time: {start_time}')
    print(f'  path_len: {path_len}, curr_idx: {curr_idx}, edge_pct: {edge_pct:.3f}')
    print(f'  completion: {completion:.3f}')
    print(f'  train_sim_time: {train_sim_time}')
    print(f'  total_travel_budget: {total_travel_budget}')
    print(f'  expected_completion: {expected_completion:.3f}')
    print(f'  delay: {delay:.3f}')
    print(f'  On time? {delay <= DELAY_THRESHOLD}')
    break
