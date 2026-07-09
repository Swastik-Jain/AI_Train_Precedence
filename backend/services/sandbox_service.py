from typing import Dict, Any
import uuid
from datetime import datetime, timezone
import numpy as np
from services import system_service
from services.simulation_service import _get_sim_brain
from fastapi import HTTPException
from services.maintenance_service import sync_blocks_to_rl_env, _resolve_reroute_strategy
from routers.websockets import broadcast_topology

def _run_forward_simulation(state, n_ticks: int, latencies: dict, forced_actions: dict, apply_sandbox_blocks: bool = True) -> dict:
    """
    Clone current live state into a throwaway env copy.
    Run N ticks. Return outcome metrics.
    Does NOT touch state.train_states or live env.
    Applies state.sandbox_blocks (what-if blocks) to the cloned env — not state.active_blocks.
    """
    model, live_vec_env = _get_sim_brain(state)
    if live_vec_env is None:
        return None

    import copy
    import numpy as np
    
    sandbox_inner = copy.deepcopy(live_vec_env.venv.envs[0])

    # Apply sandbox (what-if) blocks to the cloned env's track_map.
    # This makes the forward simulation reflect what-if constraints WITHOUT
    # touching the live inner_env or calling _sync_blocks_to_rl_env().
    if apply_sandbox_blocks and state.sandbox_blocks:
        patched_map = copy.deepcopy(sandbox_inner.track_map) if hasattr(sandbox_inner, 'track_map') else {}
        for edge_id, blk in state.sandbox_blocks.items():
            parts = edge_id.split("-")
            if len(parts) < 3:
                continue
            try:
                src = int(parts[1])
                dst = int(parts[2])
            except ValueError:
                continue

            if blk.get("severity") == "TOTAL_BLOCK":
                if src in patched_map and dst in patched_map[src].get("next", []):
                    patched_map[src]["next"] = [n for n in patched_map[src]["next"] if n != dst]
                if dst in patched_map and src in patched_map[dst].get("prev", []):
                    patched_map[dst]["prev"] = [n for n in patched_map[dst]["prev"] if n != src]
            elif blk.get("severity") == "SPEED_RESTRICTION":
                limit = blk.get("speed_limit", 30)
                if dst in patched_map:
                    patched_map[dst]["speed"] = limit

        if hasattr(sandbox_inner, 'track_map'):
            sandbox_inner.track_map = patched_map

    # Inject the delay
    active_latencies = {}
    if latencies:
        active_latencies = latencies.copy()
        for t in sandbox_inner.trains:
            t_id = t.get('id')
            if t_id in latencies:
                if t_id in sandbox_inner.schedule:
                    sandbox_inner.schedule[t_id]['deadline'] -= latencies[t_id]

    outcomes = {'finished': 0, 'total_delay_min': 0.0, 'holds': 0, 'diverts': 0, 'adjustments': [], 'projected_schedule': {}, 'holds_total': 0, 'active_train_ticks': 0, 'spawned_trains': 0, 'on_time_trains': 0}
    for t in sandbox_inner.trains:
        outcomes['projected_schedule'][t.get('id')] = {}
    
    for tick in range(n_ticks):
        obs = sandbox_inner._get_observation()
        if model:
            # Normalize single observation
            obs_batch = np.expand_dims(obs, axis=0)
            norm_obs_batch = live_vec_env.normalize_obs(obs_batch)
            actions_batch, _ = model.predict(norm_obs_batch, deterministic=True)
            actions = actions_batch[0].copy() if hasattr(actions_batch[0], 'copy') else np.array(actions_batch[0])
        else:
            actions = np.ones(sandbox_inner.observation_space.shape[0], dtype=np.int64)

        # Apply forced actions and latencies
        for idx, t in enumerate(sandbox_inner.trains):
            t_id = t.get('id')
            if t_id in forced_actions and forced_actions[t_id] in (0, 1, 2):
                actions[idx] = forced_actions[t_id]
            elif t_id in active_latencies and active_latencies[t_id] > 0:
                actions[idx] = 0
                active_latencies[t_id] -= 1
                
            # Log holds/diverts for the first tick only for adjustments
            if tick == 0 and not t.get('finished', False) and t.get('position', 0) not in (0, 998, 999):
                act = actions[idx]
                if act == 0:
                    outcomes['holds'] += 1
                    outcomes['adjustments'].append({
                        "id": len(outcomes['adjustments'])+1,
                        "type": "Signal Hold",
                        "desc": f"{t_id} held by simulation at its current block.",
                        "train_id": t_id,
                        "edge_id": "unknown",  # We can refine this if needed
                        "constraint_type": "SPEED_LIMIT",
                        "value": 0
                    })
                elif act == 2:
                    outcomes['diverts'] += 1
                    outcomes['adjustments'].append({
                        "id": len(outcomes['adjustments'])+1,
                        "type": "Spatial Reroute",
                        "desc": f"{t_id} diverted to loop.",
                        "train_id": t_id,
                        "edge_id": "unknown",
                        "constraint_type": "REROUTE",
                        "value": 2
                    })

        # Apply the OR-Shield safety checks to match live simulation behavior
        if state.or_shield_enabled:
            # optimize_decision returns safe_actions, decision_meta
            safe_actions, _ = state.or_shield.optimize_decision(
                trains=sandbox_inner.trains,
                ai_actions=actions.tolist() if hasattr(actions, 'tolist') else actions,
                track_map=sandbox_inner.track_map,
                raw_actions=actions.tolist() if hasattr(actions, 'tolist') else actions,
            )
            safe_actions = np.array(safe_actions, dtype=np.int64)
        else:
            safe_actions = np.array(actions, dtype=np.int64)
            
        for idx, t in enumerate(sandbox_inner.trains):
            if not t.get('finished', False) and t.get('position', 0) not in (0, 998, 999):
                outcomes['active_train_ticks'] += 1
                if safe_actions[idx] == 0:
                    outcomes['holds_total'] += 1
                    
        _, _, terminated, truncated, _ = sandbox_inner.step(safe_actions)
        
        # Track projected positions
        sim_tick_offset = state.sim_tick
        for t in sandbox_inner.trains:
            t_id = t.get('id')
            pos = t.get('position', 0)
            node_name = f"node-{pos}"
            if t_id in outcomes['projected_schedule']:
                if node_name not in outcomes['projected_schedule'][t_id]:
                    outcomes['projected_schedule'][t_id][node_name] = {"arrival": sim_tick_offset + tick, "departure": sim_tick_offset + tick}
                else:
                    outcomes['projected_schedule'][t_id][node_name]["departure"] = sim_tick_offset + tick

        if terminated or truncated:
            break

    for t in sandbox_inner.trains:
        sched = sandbox_inner.schedule.get(t.get('id'), {})
        # Count if train has spawned, OR if it's stuck at spawn but past its scheduled departure
        has_spawned = t.get('position', 0) != 0
        should_have_spawned = sched.get('start_time', 9999) <= sandbox_inner.sim_time
        
        if has_spawned or should_have_spawned:
            outcomes['spawned_trains'] += 1
            if t.get('finished', False):
                outcomes['finished'] += 1
            
            sched = sandbox_inner.schedule.get(t.get('id'), {})
            t_time = t.get('finish_step')
            if t_time is None:
                # Project the finish time based on remaining distance
                if hasattr(sandbox_inner, '_get_train_km') and hasattr(sandbox_inner, 'SECTION_LENGTH_KM'):
                    my_km = sandbox_inner._get_train_km(t)
                    if t.get('direction', 'DOWN') == 'DOWN':
                        km_done = my_km
                    else:
                        km_done = sandbox_inner.SECTION_LENGTH_KM - my_km
                    
                    dist_remaining_km = sandbox_inner.SECTION_LENGTH_KM - km_done
                    # Assume an average speed of 50 km/h
                    projected_remaining_mins = (dist_remaining_km / 50.0) * 60.0
                else:
                    projected_remaining_mins = 0
                
                t_time = sandbox_inner.sim_time + projected_remaining_mins
                
            current_delay = max(0, t_time - sched.get('deadline', 9999))
            
            if current_delay <= 5.0:
                outcomes['on_time_trains'] += 1
            outcomes['total_delay_min'] += current_delay

    return outcomes


def analyze_simulation(state, req):
    """
    True forward simulation: clones the live environment, injects delays/forces,
    runs for N ticks, and compares outcomes against a baseline clone.
    """
    latencies = req.latencies or {}
    forced_actions = req.forced_actions or {}

    # Allow simulation even if no live trains have spawned yet (state.train_states is empty)
    # The forward simulation will advance time and naturally spawn the upcoming scheduled trains.

    # Run baseline (no delay, no what-if blocks)
    baseline_outcomes = _run_forward_simulation(state, 120, {}, {}, apply_sandbox_blocks=False)
    
    # Run scenario (with latencies, forced actions, and what-if blocks)
    scenario_outcomes = _run_forward_simulation(state, 120, latencies, forced_actions, apply_sandbox_blocks=True)
    
    if baseline_outcomes is None or scenario_outcomes is None:
        raise HTTPException(status_code=503, detail="Simulation unavailable (model missing?).")
        
    adjustments = scenario_outcomes['adjustments']
    
    # Always add a speed-cap proposal for the delayed trains if applicable
    for d_tid, l_min in latencies.items():
        speed_cap = max(20, 90 - l_min * 2)
        # get edge from live state
        edge_id = "edge-1-2"
        for t in state.train_states.values():
            if t.get("train_id") == d_tid:
                edge_id = t.get("edge_id", "edge-1-2")
                break
                
        adjustments.insert(0, {
            "id": len(adjustments),
            "type": "Dynamic Speed Cap",
            "desc": (
                f"{d_tid} speed capped to {speed_cap} km/h to prevent "
                f"rear-end risk. Latency: +{l_min} min."
            ),
            "train_id": d_tid,
            "edge_id": edge_id,
            "constraint_type": "SPEED_LIMIT",
            "value": speed_cap
        })
        
    # Set proper edge_ids on the adjustments using live data
    for adj in adjustments:
        if adj["edge_id"] == "unknown":
            for t in state.train_states.values():
                if t.get("train_id") == adj["train_id"]:
                    adj["edge_id"] = t.get("edge_id", "unknown")
                    break

    # Reliability: On-Time Performance (Trains with <= 5 min delay)
    baseline_otp = (baseline_outcomes['on_time_trains'] / max(1, baseline_outcomes['spawned_trains'])) * 100
    scenario_otp = (scenario_outcomes['on_time_trains'] / max(1, scenario_outcomes['spawned_trains'])) * 100
    otp_diff = scenario_otp - baseline_otp
    rel_sign = "+" if otp_diff > 0 else ""
    reliability_pct = f"{scenario_otp:.1f}% OTP (Δ {rel_sign}{otp_diff:.1f}%)"
    
    # Congestion: Percentage of time trains spent stopped at red signals
    baseline_stopped = (baseline_outcomes['holds_total'] / max(1, baseline_outcomes['active_train_ticks'])) * 100
    scenario_stopped = (scenario_outcomes['holds_total'] / max(1, scenario_outcomes['active_train_ticks'])) * 100
    stopped_diff = scenario_stopped - baseline_stopped
    cong_sign = "+" if stopped_diff > 0 else ""
    congestion_pct = f"{scenario_stopped:.1f}% Stopped (Δ {cong_sign}{stopped_diff:.1f}%)"

    result = {
        "label": req.label,
        "source": "True Forward-Simulation (120 ticks)",
        "latencies_applied": latencies,
        "forced_actions_applied": list(forced_actions.keys()),
        "impact": {
            "reliability": reliability_pct,
            "congestion": congestion_pct,
            "baseline_finished": baseline_outcomes['finished'],
            "scenario_finished": scenario_outcomes['finished'],
            "baseline_delay": round(baseline_outcomes['total_delay_min'], 1),
            "scenario_delay": round(scenario_outcomes['total_delay_min'], 1)
        },
        "adjustments": adjustments
    }
    return result



async def deploy_simulation(payload: dict, state):
    """
    Deploys a set of active blocks and dynamic constraints from the Sandbox
    into the live production environment.
    """
    blocks = payload.get("blocks", [])
    forced_actions = payload.get("forced_actions", {})
    latencies = payload.get("latencies", {})

    # Apply forced_actions as sticky overrides for 60 ticks
    for t_id, action in forced_actions.items():
        state.sticky_actions[t_id] = (action, state.sim_tick + 60)
        
    # Apply latencies by reducing deadline in the live environment and holding the train
    if latencies:
        model, env = _get_sim_brain(state)
        if env:
            inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
            for t_id, lat in latencies.items():
                if t_id in inner_env.schedule:
                    inner_env.schedule[t_id]['deadline'] -= lat
                
                # If there's no explicit forced action, physically hold the train for `lat` ticks
                if t_id not in forced_actions:
                    state.sticky_actions[t_id] = (0, state.sim_tick + lat)

    # Deploy real blocks only — explicitly reject any what-if blocks that
    # may have been included by accident. isWhatIf: true blocks must never
    # enter state.active_blocks or affect live inference.
    real_blocks = [b for b in blocks if not b.get("isWhatIf", False)]
    skipped = len(blocks) - len(real_blocks)
    if skipped > 0:
        print(f"[SANDBOX] ⚠️  Skipped {skipped} what-if block(s) during deploy — they must not enter state.active_blocks")

    for b in real_blocks:
        element_id = b.get("element_id")
        if not element_id:
            continue

        block_dict = {
            "blockId": b.get("blockId", str(uuid.uuid4())),
            "element_id": element_id,
            "type": b.get("type", "TRACK_FAULT"),
            "severity": b.get("severity", "TOTAL_BLOCK"),
            "reason": b.get("reason", "Deployed from Sandbox"),
            "applied_at": system_service._now_iso(),
            "isWhatIf": False,
            # Preserve start_time/end_time if provided so time-gating works
            "start_time": b.get("start_time", system_service._now_iso()),
            "end_time": b.get("end_time", system_service._now_iso()),
        }

        state.active_blocks[element_id] = block_dict
        sync_blocks_to_rl_env(state)  # Immediately patch live RL model
        impact = _resolve_reroute_strategy(state, element_id)

        await broadcast_topology({
            "type": "MAINTENANCE_BLOCK_APPLIED",
            "block": block_dict,
            "impact": impact,
        })
    
    constraints_count = len(forced_actions) + len(latencies)
    # Audit Log
    system_service.push_audit_log(state, {
        "t": system_service._now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SIMULATION_SANDBOX",
        "action": f"Deployed {len(real_blocks)} real block(s) and {constraints_count} constraint(s). {skipped} what-if block(s) skipped.",
        "operator": "Chief Dispatcher",
        "status": "Deployed",
        "statusType": "success",
        "id": str(uuid.uuid4())
    })

    print(f"[SANDBOX] 🚀 Simulation Deployed! ({len(real_blocks)} real blocks, {constraints_count} constraints, {skipped} what-if skipped)")
    return {"status": "success", "blocks_applied": len(real_blocks), "constraints_applied": constraints_count, "whatif_skipped": skipped}


def get_impact_analysis(state):
    """
    Return a consolidated impact report for ALL currently active blocks.
    Used by the frontend to trigger the 'Ripple Effect' notification.
    """
    if not state.active_blocks:
        return {
            "status": "clear",
            "message": "No active maintenance blocks. Network operating normally.",
            "total_affected_trains": 0,
            "blocks": [],
            "timestamp": system_service._now_iso(),
        }

    reports = []
    total_affected: set = set()

    for element_id in state.active_blocks:
        report = _resolve_reroute_strategy(state, element_id)
        reports.append(report)
        total_affected.update(report.get("affected_train_ids", []))

    # Overall summary message
    n = len(total_affected)
    strategies = list({r["strategy"] for r in reports})
    strategy_str = " / ".join(s.replace("_", " ").title() for s in strategies)

    return {
        "status": "blocks_active",
        "total_affected_trains": n,
        "affected_train_ids": list(total_affected),
        "primary_strategy": reports[0]["strategy"] if reports else "TEMPORAL_SHIFT",
        "message": (
            f"Maintenance on {len(state.active_blocks)} segment(s) affects {n} upcoming train(s). "
            f"Rerouting strategy: {strategy_str}."
        ),
        "block_reports": reports,
        "timestamp": system_service._now_iso(),
    }

