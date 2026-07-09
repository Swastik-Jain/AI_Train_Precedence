import random
from typing import Dict, Any
import config
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
            
        dir_str = "DOWN" if dir_val in (1, "DOWN") else "UP"
        if dir_str == "UP":
            incoming_trains += 1
        else:
            outgoing_trains += 1
        
        if t_state.get("status") in ("Blocked", "Halted"):
            halted_trains += 1

    # ── Punctuality Calculation ──────────────────────────────────────────────
    # The RL environment (train_env.py) tracks time in sim_time (steps since
    # episode start). state.sim_tick mirrors this — both start at 0 and
    # increment by 1 each simulation step.
    # Deadlines are in the same sim_time unit (set in ai/config.py).
    #
    # For FINISHED trains: compare finish_step vs deadline directly.
    # For RUNNING trains:  estimate whether the train can make its deadline
    #                      by looking at progress (edge position within the
    #                      84-node corridor) scaled to remaining time.
    # ────────────────────────────────────────────────────────────────────────
    DELAY_THRESHOLD = 15.0          # ticks of tolerance (generous buffer)
    on_time_trains = 0
    evaluated_trains = 0

    for t_id, t_state in state.train_states.items():
        status = t_state.get("status", "Scheduled")
        if status == "Scheduled":
            continue

        reg = state.fleet_registry.get(t_id, {})
        deadline   = reg.get("deadline", 9999)   # sim-ticks, same unit as sim_tick
        start_time = reg.get("start_time", 0)

        delay = 0.0

        if status == "Finished" or t_state.get("finished", False):
            # --- Finished trains: use locked-in finish tick ---
            t_actual = t_state.get("finish_step")
            if t_actual is None:
                t_actual = t_state.get("finish_time")
            if t_actual is None:
                # Truly unknown — assume on-time to avoid penalising prematurely
                t_actual = deadline
            delay = max(0.0, float(t_actual) - float(deadline))

        else:
            # --- Running trains: progress-based estimation ---
            # Nodes 0..83 → edge-{n}-{n+1}, node 83 → edge-83-999
            # Map current edge to a node index [0..83]
            curr_edge = t_state.get("edge_id", "")
            node_progress = None
            try:
                if curr_edge and curr_edge.startswith("edge-"):
                    parts = curr_edge.split("-")
                    if len(parts) >= 3:
                        src_node = int(parts[1])
                        edge_pct = t_state.get("position_percentage", 0.0)
                        # Node index along 84-node corridor (0..83)
                        node_progress = src_node + edge_pct
            except Exception:
                pass

            if node_progress is not None:
                direction_raw = reg.get("direction", t_state.get("direction", "DOWN"))
                direction_str = "DOWN" if direction_raw in (1, "DOWN") else "UP"

                # Nodes go 0→83 for DOWN, 0→83 means reverse for UP (starts at 83)
                # Fraction of journey COMPLETED (0.0=just started, 1.0=done)
                TOTAL_NODES = 84.0
                if direction_str == "DOWN":
                    frac_done = node_progress / TOTAL_NODES
                else:
                    # UP trains travel from node 83→0 through the corridor
                    # Their position comes in as source node of the edge they
                    # are currently on.  A freshly-started UP train is near 83.
                    frac_done = (TOTAL_NODES - node_progress) / TOTAL_NODES

                frac_done = max(0.0, min(frac_done, 1.0))
                frac_remaining = 1.0 - frac_done

                # Time elapsed since train started, and remaining budget
                # Use inference_sim_time (episodic, resets to 0 each episode)
                # NOT sim_tick (global wall clock that never resets).
                ep_time        = getattr(state, 'inference_sim_time', state.sim_tick)
                elapsed        = ep_time - start_time
                total_budget   = deadline - start_time          # ticks allocated
                time_remaining = deadline - ep_time

                if frac_done < 0.01:
                    # Train just spawned — give it the benefit of the doubt
                    delay = 0.0
                elif time_remaining <= 0:
                    # Already past deadline and still running
                    delay = float(-time_remaining) + frac_remaining * (total_budget * 0.5)
                elif elapsed > 0 and frac_done > 0:
                    # Extrapolate: at current pace, when will train finish?
                    ticks_per_frac   = elapsed / frac_done
                    estimated_finish = ep_time + (frac_remaining * ticks_per_frac)
                    delay = max(0.0, estimated_finish - deadline)
                else:
                    delay = 0.0
            else:
                # Edge info unavailable — assume on-time
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
