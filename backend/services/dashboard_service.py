import random
from typing import Dict, Any
import config
from ai.config import SECTION_LENGTH_KM
from services import system_service
from services import maintenance_service


def get_telemetry(state):
    """Returns real-time telemetry calculated from the current simulation state."""
    active_trains = 0
    incoming_trains = 0
    outgoing_trains = 0
    terminal_trains = 0
    halted_trains = 0

    for t_id, t_state in state.train_states.items():
        if t_state.get("status") == "Scheduled":
            continue
        if t_state.get("status") == "Finished":
            terminal_trains += 1
            continue
            
        active_trains += 1
        
        # Determine direction based on state
        dir_val = t_state.get("direction")
        if dir_val is None:
            dir_val = state.fleet_registry.get(t_id, {}).get("direction", "DOWN")
            
        if dir_val == "UP" or dir_val == 1:
            incoming_trains += 1
        else:
            outgoing_trains += 1
        
        if t_state.get("status") in ("Blocked", "Halted"):
            halted_trains += 1

    DELAY_THRESHOLD = 10.0
    on_time_trains = 0
    evaluated_trains = 0

    for t_id, t_state in state.train_states.items():
        if t_state.get("status") == "Scheduled":
            continue
            
        reg = state.fleet_registry.get(t_id, {})
        deadline = reg.get("deadline", 120)
        start_time = reg.get("start_time", 0)

        delay = 0.0
        if t_state.get("status") == "Finished":
            t_actual = t_state.get("finish_time", state.sim_tick)
            delay = max(0.0, t_actual - deadline)
        else:
            curr_edge = t_state.get("edge_id")
            if curr_edge and curr_edge.startswith("edge-"):
                try:
                    parts = curr_edge.split("-")
                    if len(parts) >= 3:
                        source_node = int(parts[1])
                        target_node = int(parts[2])
                        source_km = state.raw_track_map.get(source_node, {}).get("km", 0.0)
                        target_km = state.raw_track_map.get(target_node, {}).get("km", 0.0)
                        edge_pct = t_state.get("position_percentage", 0.0)
                        current_km = source_km + edge_pct * (target_km - source_km)
                        
                        direction = reg.get("direction", "DOWN")
                        if direction == "DOWN":
                            dist_remaining_km = max(0.0, SECTION_LENGTH_KM - current_km)
                        else:
                            dist_remaining_km = max(0.0, current_km)
                        
                        train_sim_time = state.sim_tick
                        time_remaining_budget = deadline - train_sim_time
                        
                        if dist_remaining_km <= 0:
                            delay = 0.0
                        elif time_remaining_budget <= 0:
                            # Already late, estimate 1 min per km (60 km/h) for remaining
                            delay = -time_remaining_budget + dist_remaining_km * 1.0
                        else:
                            # Assuming avg speed 60km/h (1 min per km) for remainder
                            expected_time_to_finish = dist_remaining_km * 1.0
                            estimated_finish = train_sim_time + expected_time_to_finish
                            delay = max(0.0, estimated_finish - deadline)
                except Exception as e:
                    print(f"[WARN] Failed to calculate delay for train {t_id}: {e}")
                    delay = 0.0

        if delay <= DELAY_THRESHOLD:
            on_time_trains += 1
            
        evaluated_trains += 1

    if evaluated_trains > 0:
        punctuality = (on_time_trains / evaluated_trains) * 100.0
        state.last_punctuality = punctuality
    else:
        punctuality = state.last_punctuality

    halted_pct = (halted_trains / active_trains * 100) if active_trains > 0 else 0
    active_blocks_count = sum(1 for b in state.active_blocks.values() if maintenance_service.is_block_active(b))
    blocks_active = active_blocks_count > 0

    if halted_pct > 20:
        network_fluidity = "Degraded"
    elif blocks_active or (10 <= halted_pct <= 20):
        network_fluidity = "Warning"
    else:
        network_fluidity = "Nominal"
    ai_load = 40 + (active_trains * 5) + (active_blocks_count * 15)
    ai_load = max(0, min(100, ai_load))

    node_response_time = random.randint(8, 12)
    if ai_load > 85:
        node_response_time = random.randint(50, 75)

    return {
        "punctuality": round(punctuality, 1),
        "active_trains": active_trains,
        "incoming_trains": incoming_trains,
        "outgoing_trains": outgoing_trains,
        "terminal_trains": terminal_trains,
        "network_fluidity": network_fluidity,
        "halted_pct": round(halted_pct, 1),
        "halted_trains": halted_trains,
        "node_response_time": node_response_time,
        "ai_load": ai_load,
        "schedule_ready": len(state.last_or_schedule) > 0,
        "schedule_train_count": len(state.last_or_schedule),
        "lockdown": state.system_lockdown,
        "active": state.inference_active,
        "timestamp": system_service._now_iso()
    }

# ---------------------------------------------------------------------------
# WebSockets
# ---------------------------------------------------------------------------
