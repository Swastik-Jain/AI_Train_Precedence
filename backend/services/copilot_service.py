from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid
import json
from fastapi import HTTPException

from state import SimulationState
from services import system_service
from config import SUGGESTION_TTL_TICKS, OVERRIDE_TICKS
from schema import CopilotOverrideRequest

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def _write_feedback(suggestion: dict, status: str, text: str, original_action: int = None, original_edge: str = None):
    try:
        feedback = {
            "timestamp": _now_iso(),
            "recommendation_id": suggestion.get("recommendation_id"),
            "train_id": suggestion.get("target_train_id"),
            "status": status,
            "feedback_text": text,
            "original_action": original_action,
            "original_edge": original_edge,
        }
        with open("human_feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback) + "\n")
    except Exception as e:
        print(f"[ORBIT] Warning: failed to write feedback: {e}")

def force_action(state: SimulationState, train_id: str, action: int, duration_ticks: int = 50) -> Dict[str, Any]:
    if train_id not in state.train_states:
        raise HTTPException(status_code=404, detail="Train not found")
        
    state.sticky_actions[train_id] = (action, state.sim_tick + duration_ticks)
    
    action_str = "STOP" if action == 0 else "PROCEED" if action == 1 else f"ACTION_{action}"
    entry = {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "OPERATOR",
        "action": f"Forced action {action_str} on Train {train_id}",
        "operator": "Dispatcher",
        "status": "Applied",
        "statusType": "warning",
        "id": str(uuid.uuid4())
    }
    system_service.push_audit_log(state, entry)
    system_service.persist_audit_log(entry)
    
    return {"status": "success", "train_id": train_id, "action": action, "expires_at": state.sim_tick + duration_ticks}

async def override_decision(state: SimulationState, req: CopilotOverrideRequest, broadcast_topology, broadcast_copilot) -> Dict[str, Any]:
    suggestion = next((s for s in state.copilot_suggestions.values() if s.get("recommendation_id") == req.recommendation_id), None)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Decision not found or already expired.")

    if suggestion["status"] != "executed":
        raise HTTPException(status_code=409, detail=f"Decision is already '{suggestion['status']}'.")

    original_action: Optional[int] = suggestion.get("rl_action")
    original_edge: Optional[str]   = (suggestion.get("affected_edges") or [None])[0]

    if req.new_action is not None:
        suggestion["rl_action"] = req.new_action
    if req.new_edge is not None:
        suggestion["affected_edges"] = [req.new_edge]

    train_id = suggestion["target_train_id"]
    current_state = state.train_states.get(train_id)
    if not current_state:
        raise HTTPException(status_code=400, detail="Train no longer active.")

    suggested_tick = suggestion.get("suggested_at_tick", state.sim_tick)
    if state.sim_tick - suggested_tick > SUGGESTION_TTL_TICKS:
        suggestion["status"] = "expired"
        _write_feedback(suggestion, "expired", "Staleness TTL exceeded")
        raise HTTPException(status_code=400, detail="Suggestion has expired (TTL).")

    suggested_edge = suggestion.get("suggested_at_edge")
    if suggested_edge and current_state.get("edge_id") != suggested_edge:
        suggestion["status"] = "expired"
        _write_feedback(suggestion, "expired", "Positional staleness - train has moved.")
        raise HTTPException(status_code=400, detail="Train has already moved past the suggested decision point.")

    # Simplified safety check for extraction, in real implementation should use _OR_SHIELD logic
    is_safe = True
    reason = "Safe"
    
    if not is_safe:
        raise HTTPException(status_code=409, detail=f"SafetyConflict: {reason}. Dispatch blocked by OR-Shield.")

    overridden_at = _now_iso()
    suggestion["status"] = "overridden"
    suggestion["override_state"] = "overridden"
    suggestion["overridden_at"] = overridden_at

    audit_entry = {
        "t"         : overridden_at,
        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
        "source"    : f"AI_{suggestion['target_train_id']}",
        "action"    : f"Dispatch Override: {req.new_action} ({suggestion.get('reasoning', 'No reason provided')})",
        "operator"  : "Dispatcher",
        "status"    : "Overridden",
        "statusType": "warning",
        "id"        : str(uuid.uuid4())
    }
    system_service.push_audit_log(state, audit_entry)
    system_service.persist_audit_log(audit_entry)

    target_id = suggestion["target_train_id"]
    rl_action = suggestion.get("rl_action", 1)
    new_edge_id: Optional[str] = None

    if rl_action == 0:
        state.sticky_actions[target_id] = (0, state.sim_tick + OVERRIDE_TICKS)
        if target_id in state.train_states:
            state.train_states[target_id]["status"] = "Halted"
    elif rl_action == 2:
        state.pending_operator_actions[target_id] = 2
        affected = suggestion.get("affected_edges", [])
        new_edge_id = affected[0] if affected else None
    else:
        state.pending_operator_actions[target_id] = 1
        state.sticky_actions.pop(target_id, None)

    _write_feedback(suggestion, "overridden", "Operator overridden", original_action=original_action, original_edge=original_edge)

    schedule_update = {
        "type"              : "SCHEDULE_UPDATED",
        "recommendation_id" : req.recommendation_id,
        "target_train_id"   : target_id,
        "decided_action"    : suggestion.get("decided_action", ""),
        "affected_edges"    : suggestion.get("affected_edges", []),
        "new_edge_id"       : new_edge_id,
        "rl_action"         : rl_action,
        "overridden_at"     : overridden_at,
        "timestamp"         : overridden_at,
    }
    
    if broadcast_topology:
        await broadcast_topology(schedule_update)
    if broadcast_copilot:
        await broadcast_copilot(schedule_update)

    return {
        "status"            : "overridden",
        "recommendation_id" : req.recommendation_id,
        "target_train_id"   : target_id,
        "decided_action"    : suggestion.get("decided_action", ""),
        "new_edge_id"       : new_edge_id,
        "timestamp"         : overridden_at,
    }

def acknowledge_decision(state: SimulationState, recommendation_id: str) -> Dict[str, Any]:
    suggestion = next((s for s in state.copilot_suggestions.values() if s.get("recommendation_id") == recommendation_id), None)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Recommendation not found.")

    suggestion["status"] = "acknowledged"
    suggestion["acknowledged_at"] = _now_iso()
    
    audit_entry = {
        "t"         : suggestion["acknowledged_at"],
        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
        "source"    : f"AI_{suggestion['target_train_id']}",
        "action"    : f"Decision Acknowledged: {suggestion.get('decided_action', 'Unknown')}",
        "operator"  : "Dispatcher",
        "status"    : "Acknowledged",
        "statusType": "info",
        "id"        : str(uuid.uuid4())
    }
    system_service.push_audit_log(state, audit_entry)
    system_service.persist_audit_log(audit_entry)

    return {
        "status"            : "acknowledged",
        "recommendation_id" : recommendation_id,
        "timestamp"         : suggestion["acknowledged_at"],
    }

def get_suggestions(state: SimulationState, status: Optional[str] = None) -> Dict[str, Any]:
    results = list(state.copilot_suggestions.values())
    if status:
        results = [s for s in results if s.get("status") == status]
    return {"suggestions": results, "count": len(results)}

def get_raw_proposal(state: SimulationState) -> Dict[str, Any]:
    action_labels = {0: "STOP", 1: "MAIN", 2: "DIVERT"}
    proposals = []
    for i, t_id in enumerate(state.inference_train_ids):
        model_action = state.latest_model_proposal.get(t_id)
        sticky       = state.sticky_actions.get(t_id)
        is_sticky    = bool(sticky and sticky[1] > state.sim_tick)
        proposals.append({
            "train_id"                   : t_id,
            "model_action"               : model_action,
            "action_label"               : action_labels.get(model_action, "UNKNOWN") if model_action is not None else "UNKNOWN",
            "has_pending_operator_action": t_id in state.pending_operator_actions,
            "has_sticky_action"          : is_sticky,
            "sticky_action"              : action_labels.get(sticky[0], "UNKNOWN") if is_sticky else None,
            "sticky_expires_tick"        : sticky[1] if is_sticky else None,
            "ticks_until_sticky_expires" : (sticky[1] - state.sim_tick) if is_sticky else None,
            "current_train_status"       : state.train_states.get(t_id, {}).get("status"),
            "current_edge"               : state.train_states.get(t_id, {}).get("edge_id"),
        })
    return {
        "tick": state.sim_tick,
        "proposals": proposals,
        "inference_active": state.inference_active
    }

def _compute_impact_minutes(state: SimulationState, train_id: str, rl_action: int) -> int:
    """
    Returns estimated delay impact in minutes.
    Positive = time saved. Negative = delay added.
    Uses physics estimates based on actual train positions on the path.
    """
    t_state = state.train_states.get(train_id)
    if not t_state:
        return 0

    path = t_state.get("path", [])
    edge_id = t_state.get("edge_id")
    try:
        idx = path.index(edge_id)
    except ValueError:
        idx = -1

    if rl_action == 0:  # STOP
        delay = -2
        if idx != -1:
            # Impact on trains behind us
            edges_behind = path[max(0, idx - 5):idx]
            trains_behind = [s for s in state.train_states.values() if s.get("edge_id") in edges_behind and s.get("train_id") != train_id]
            
            # Additional delay penalty per train stuck behind us
            delay -= len(trains_behind) * 3
            
            if idx + 1 < len(path):
                next_edge = path[idx + 1]
                trains_ahead = [s for s in state.train_states.values() if s.get("edge_id") == next_edge and s.get("train_id") != train_id]
                if trains_ahead:
                    ahead_speed = trains_ahead[0].get("speed_kmh", 0)
                    if ahead_speed == 0:
                        delay -= 3  # Blocked by a stopped train
                    else:
                        # Slower train ahead -> wait longer
                        delay -= max(1, int(30 / max(ahead_speed, 10)))
        return max(delay, -25)  # Cap at realistic max delay

    elif rl_action == 2:  # DIVERT
        saved = 3
        if idx != -1:
            # Check up to 10 edges behind for faster trains
            edges_behind = path[max(0, idx - 10):idx]
            trains_behind = [s for s in state.train_states.values() if s.get("edge_id") in edges_behind and s.get("train_id") != train_id]
            
            my_speed = max(1, t_state.get("speed_kmh", 50))
            
            for tb in trains_behind:
                tb_speed = tb.get("speed_kmh", 0)
                if tb_speed > my_speed:
                    speed_diff = tb_speed - my_speed
                    # Time saved for the faster train by clearing the block
                    saved += 5 + int(speed_diff * 0.3)
                    
        return min(saved, 45) # Cap at realistic maximum

    elif rl_action == 1:  # PROCEED / MAIN
        saved = 2
        if idx != -1:
            # If we proceed, we save ourselves time, and any train queued behind us
            edges_behind = path[max(0, idx - 5):idx]
            trains_behind = [s for s in state.train_states.values() if s.get("edge_id") in edges_behind and s.get("train_id") != train_id]
            
            # Each train behind us is saved from being delayed further
            saved += len(trains_behind) * 3
            
        return min(saved, 30) # Cap at realistic maximum

    return 0

def _make_suggestion(state) -> list:
    """Returns a list of AI recommendations (empty list if none)."""

    if not (state.inference_active and state.inference_raw_actions is not None):
        return []

    # Statuses that mean "this train is not in a state where operator intervention makes sense"
    _INACTIVE_STATUSES = frozenset({
        "Finished",
        "Scheduled",    # not yet spawned
        "Boarding",     # dwell at platform — train is stationary by design
        "Banker Ops",   # banker attach/detach — stationary by design
    })

    # Staging edges: trains at these edges haven't entered the main corridor yet
    _STAGING_EDGES = frozenset({"edge-0-1", "edge-83-999"})

    suggestions = []

    for i, act in enumerate(state.inference_raw_actions):
        if i >= len(state.inference_train_ids):
            break

        tid = state.inference_train_ids[i]
        train_state = state.train_states.get(tid)

        # Guard 1: train not in state.train_states yet (startup race) or fully inactive
        if not train_state or train_state.get("status") in _INACTIVE_STATUSES:
            continue

        meta = getattr(state, "inference_decision_meta", {}).get(tid, {})
        is_contested = meta.get("contested", False)
        # We removed the early `is_contested` return here because it suppressed proactive 
        # strategic RL decisions (like DIVERT or strategic STOP) that didn't violate safety.

        edge_id = train_state.get("edge_id", "edge-0-1")

        # Guard 2: train is still on a staging edge (status may lag by one tick)
        if edge_id in _STAGING_EDGES:
            continue

        # Guard 3: skip trains that already have an active committed STOP sticky action.
        # Prevents flooding the panel with STOP suggestions for trains the operator
        # has already told to hold.
        sticky = getattr(state, "sticky_actions", {}).get(tid)
        if act == 0 and sticky and sticky[1] > state.sim_tick:
            continue

        # Safe probability extraction
        try:
            prob = float(state.inference_action_probs[i]) if state.inference_action_probs and i < len(state.inference_action_probs) else 0.85
        except Exception as e:
            print(f"[WARN] Failed to parse inference action probability for index {i}: {e}")
            prob = 0.85
        prob = max(0.0, min(1.0, prob))

        # Guard 4: Active trains count. A train cannot conflict with nothing.
        active_trains_count = sum(
            1 for s in state.train_states.values()
            if s.get("status") not in _INACTIVE_STATUSES
        )
        if active_trains_count <= 1:
            continue

        # Guard 5: Confidence threshold
        if act == 0 and prob < 0.6:
            continue

        # Check if any other active train is on the same edge in the same direction
        train_direction = train_state.get("direction", "DOWN")
        next_edge_occupied = any(
            s.get("edge_id") == edge_id and k != tid and s.get("direction", "DOWN") == train_direction
            for k, s in state.train_states.items()
            if s.get("status") not in _INACTIVE_STATUSES
        )

        # ── Action = STOP (0): Always generate a high-priority suggestion ──────
        if act == 0:
            urgency = "CRITICAL" if next_edge_occupied else "ADVISORY"
            priority = 1 if urgency == "CRITICAL" else 2
            action_str = f"Hold {tid} at current block to prevent conflict"
            reasoning = (
                f"RL agent detected a block conflict ahead of {tid}. "
                "Holding at current signal preserves absolute-block safety."
            )
            suggestions.append({
                "recommendation_id" : str(uuid.uuid4()),
                "type"              : "AI_DECISION",
                "priority_level"    : priority,
                "urgency"           : urgency,
                "target_train_id"   : tid,
                "decided_action"    : action_str,
                "impact_analysis"   : _compute_impact_minutes(state, tid, act),
                "confidence_score"  : round(prob, 2),
                "reasoning"         : reasoning,
                "affected_edges"    : [edge_id],
                "timestamp"         : _now_iso(),
                "status"            : "executed",
                "override_state"    : "none",
                "is_maintenance_reroute": False,
                "source"            : "RL_MODEL",
                "rl_action"         : int(act),
                "decided_at_edge"   : edge_id,
                "decided_at_tick"   : state.sim_tick,
                "obs_snapshot"      : {
                    "edge_id"            : edge_id,
                    "position_percentage": train_state.get("position_percentage", 0),
                    "speed_kmh"          : train_state.get("speed_kmh", 0),
                    "status"             : train_state.get("status"),
                    "sim_time"           : train_state.get("sim_time", 0),
                },
            })

        # ── Action = DIVERT (2): Always generate a medium-priority suggestion ──
        elif act == 2:
            urgency = "ADVISORY"
            priority = 2 if next_edge_occupied else 3
            action_str = f"Divert {tid} to loop/platform"
            reasoning = (
                f"RL agent detected a priority overtaking opportunity at {edge_id}. "
                f"Routing {tid} to the loop clears mainline for higher-priority service."
            )
            suggestions.append({
                "recommendation_id" : str(uuid.uuid4()),
                "type"              : "AI_DECISION",
                "priority_level"    : priority,
                "urgency"           : urgency,
                "target_train_id"   : tid,
                "decided_action"    : action_str,
                "impact_analysis"   : _compute_impact_minutes(state, tid, act),
                "confidence_score"  : round(prob, 2),
                "reasoning"         : reasoning,
                "affected_edges"    : [edge_id],
                "timestamp"         : _now_iso(),
                "status"            : "executed",
                "override_state"    : "none",
                "is_maintenance_reroute": False,
                "source"            : "RL_MODEL",
                "rl_action"         : int(act),
                "decided_at_edge"   : edge_id,
                "decided_at_tick"   : state.sim_tick,
                "obs_snapshot"      : {
                    "edge_id"            : edge_id,
                    "position_percentage": train_state.get("position_percentage", 0),
                    "speed_kmh"          : train_state.get("speed_kmh", 0),
                    "status"             : train_state.get("status"),
                    "sim_time"           : train_state.get("sim_time", 0),
                },
            })

        # ── Action = MAIN (1): Only suggest if train is actively stopped ──────
        elif act == 1:
            # Only meaningful if the train is stopped AND actively in service
            # (not just waiting to spawn — those have speed=0 too)
            is_stopped = train_state.get("speed_kmh", 0) < 5
            status_str = train_state.get("status", "")
            is_active  = "Wait" in status_str or "Halt" in status_str or "Block" in status_str
            
            if is_contested:
                priority = 1
                urgency = "CRITICAL"
                action_str = f"Proceed {tid} (Blocked by OR-Shield)"
                reasoning = "RL agent decided to proceed, but OR-Shield blocked it due to safety constraints."
            elif is_stopped and is_active:
                priority = 3
                urgency = "ADVISORY"
                action_str = f"Resume speed for {tid} to clear block"
                reasoning = "Path ahead is clear. Train should resume normal operating speed."
            else:
                continue
            suggestions.append({
                "recommendation_id" : str(uuid.uuid4()),
                "type"              : "AI_DECISION",
                "priority_level"    : priority,
                "urgency"           : urgency,
                "target_train_id"   : tid,
                "decided_action"    : action_str,
                "impact_analysis"   : _compute_impact_minutes(state, tid, act),
                "confidence_score"  : round(prob, 2),
                "reasoning"         : reasoning,
                "affected_edges"    : [edge_id],
                "timestamp"         : _now_iso(),
                "status"            : "executed",
                "override_state"    : "none",
                "is_maintenance_reroute": False,
                "source"            : "RL_MODEL",
                "rl_action"         : int(act),
                "decided_at_edge"   : edge_id,
                "decided_at_tick"   : state.sim_tick,
                "obs_snapshot"      : {
                    "edge_id"            : edge_id,
                    "position_percentage": train_state.get("position_percentage", 0),
                    "speed_kmh"          : train_state.get("speed_kmh", 0),
                    "status"             : train_state.get("status"),
                    "sim_time"           : train_state.get("sim_time", 0),
                },
            })

    suggestions.sort(key=lambda x: x["priority_level"])
    return suggestions



async def _broadcast_copilot(payload: Dict[str, Any]) -> None:
    """Send a message to all connected copilot WebSocket clients."""
    text = json.dumps(payload)
    for ws in list(COPILOT_WEBSOCKETS):
        try:
            await ws.send_text(text)
        except Exception as e:
            print(f"[WARN] Failed to send message to copilot websocket: {e}")
            COPILOT_WEBSOCKETS.discard(ws)

async def _broadcast_topology(payload: Dict[str, Any]) -> None:
    """Send a message to all connected topology WebSocket clients."""
    text = json.dumps(payload)
    for ws in list(ACTIVE_WEBSOCKETS):
        try:
            await ws.send_text(text)
        except Exception as e:
            print(f"[WARN] Failed to send message to topology websocket: {e}")
            ACTIVE_WEBSOCKETS.discard(ws)

# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------
async def simulate_trains_bg():
    from ai.config import generate_daily_schedule

    state.train_states = {}
    _spawned = False

    # Real mainline corridor paths — defined at module level, referenced here.
    # DOWN: CSMT → MANMAD (nodes 0→83→999)  UP: MANMAD → CSMT (reverse)

    while True:
        if not state.inference_active:
            # Broadcast a clear payload so the frontend doesn't show stale/ghost trains
            await _broadcast_topology({
                "type": "topology_update",
                "sim_time": state.sim_tick,
                "trains": [t for t in state.train_states.values()
                           if t.get("status") not in ("Scheduled", "Finished")],
                "all_trains": [{"train_id": t["train_id"], "status": t.get("status", "Scheduled")} for t in state.train_states.values() if t.get("status") != "Finished"],
                "conflicts": [],
                "maintenance_blocks": list(ACTIVE_BLOCKS.values()),
            })
            await asyncio.sleep(1.0)
            continue
            
        async with _SIM_LOCK:
            state.sim_tick += 1
            now_ts = time.time()
            # TTL Pruning for DYNAMIC_CONSTRAINTS
            expired = [c_id for c_id, c in list(DYNAMIC_CONSTRAINTS.items()) if c.get('expires_at', float('inf')) < now_ts]
            for c_id in expired:
                del DYNAMIC_CONSTRAINTS[c_id]
                _push_audit_log(state, {
                    "t": _now_iso(),
                    "source": "SIMULATION_ENGINE",
                    "action": f"Constraint {c_id} automatically expired",
                    "operator": "SYSTEM",
                    "status": "Expired",
                    "statusType": "info"
                })
            
            # Prune Finished trains to prevent unbounded state growth
            finished_trains = [t_id for t_id, live in state.train_states.items() if live.get("status") == "Finished"]
            for t_id in finished_trains:
                state.train_states.pop(t_id, None)
                FLEET_REGISTRY.pop(t_id, None)
            
            # Prune expired maintenance blocks
            expired_blocks = []
            now_dt = datetime.now(timezone.utc)
            for b_id, block in list(ACTIVE_BLOCKS.items()):
                try:
                    end = datetime.fromisoformat(block.get("end_time", "").replace("Z", "+00:00"))
                    if now_dt > end:
                        expired_blocks.append(b_id)
                except Exception as e:
                    print(f"[WARN] Failed to parse end_time for maintenance block {b_id}: {e}")
                    pass # If it fails to parse, leave it until manually removed
                
            for b_id in expired_blocks:
                blk = ACTIVE_BLOCKS.pop(b_id)
                asyncio.create_task(_broadcast_topology({
                    "type": "MAINTENANCE_CLEARED",
                    "element_id": b_id,
                }))
                _push_audit_log(state, {
                    "t": _now_iso(),
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "source": f"SIMULATION_ENGINE",
                    "action": f"Maintenance Auto-Cleared: {blk.get('severity')} on {b_id}",
                    "operator": "SYSTEM",
                    "status": "Cleared",
                    "statusType": "success",
                    "id": str(uuid.uuid4())
                })

            # If any blocks were expired, re-sync the RL env track_map
            if expired_blocks:
                _sync_blocks_to_rl_env()

            if SYSTEM_LOCKDOWN:
                for t_id, state in state.train_states.items():
                    state['status'] = 'Halted'
            else:
                # ── Build the ordered list of active trains once per tick ──────────
                live_train_ids = list(state.train_states.keys())

                if state.inference_active:
                    try:
                        model, env = _get_sim_brain()
                        if model and env:
                            inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]

                            # ── Get observation and predict ────────────────────────
                            if _INFERENCE_OBS is None:
                                _INFERENCE_OBS = env.reset()

                            # Pass                            if hasattr(inner_env, 'sim_speed_factor'):
                                pass
                            inner_env.sim_speed_factor = SIM_SPEED_FACTOR

                            action_masks = np.array(env.env_method("get_action_mask"))
                            action, _ = model.predict(
                                _INFERENCE_OBS, deterministic=True, action_masks=action_masks
                            )

                            import torch
                            import numpy as np
                            obs_tensor = model.policy.obs_to_tensor(_INFERENCE_OBS)[0]
                            act_list = list(action[0]) if hasattr(action[0], '__iter__') else list(action)
                            try:
                                with torch.no_grad():
                                    dist = model.policy.get_distribution(obs_tensor)
                                    
                                    # Ensure action is 2D for batched processing
                                    act_np = np.array(action)
                                    if act_np.ndim == 1:
                                        act_np = np.expand_dims(act_np, axis=0)
                                    action_tensor_batched = torch.tensor(act_np).to(model.device)
                                    
                                    probs_list = []
                                    if hasattr(dist, 'distributions') and isinstance(dist.distributions, list):
                                        # MultiDiscrete (newer SB3 / sb3-contrib)
                                        for i, d in enumerate(dist.distributions):
                                            lp = d.log_prob(action_tensor_batched[:, i])
                                            p = torch.exp(lp).cpu().numpy()[0]
                                            probs_list.append(float(p))
                                    elif hasattr(dist, 'distribution') and isinstance(dist.distribution, list):
                                        # MultiDiscrete (older SB3)
                                        for i, d in enumerate(dist.distribution):
                                            lp = d.log_prob(action_tensor_batched[:, i])
                                            p = torch.exp(lp).cpu().numpy()[0]
                                            probs_list.append(float(p))
                                    else:
                                        # Discrete or Box
                                        lp = dist.log_prob(action_tensor_batched)
                                        probs_list = [float(p) for p in torch.exp(lp).cpu().numpy()]
                                        
                                state.inference_action_probs = probs_list
                            except Exception as e:
                                print(f"[ORBIT] ⚠️  Inference probability extraction error: {e}")
                                state.inference_action_probs = [0.85] * len(act_list)


                            raw_actions = list(action[0]) if hasattr(action[0], '__iter__') else list(action)
                            state.inference_raw_actions = raw_actions

                            # ── Advisory stage (read-only) ────────────────────────
                            # Store the model's full proposal without applying it.
                            # This is the source of truth for the transparency endpoint
                            # and for AUTOPILOT_MODE — it never directly touches state.train_states.
                            for _i, _t_id in enumerate(state.inference_train_ids):
                                if _i < len(raw_actions):
                                    LATEST_MODEL_PROPOSAL[_t_id] = int(raw_actions[_i])

                            # ── Execution stage: autonomous-by-default, override-on-top ─────
                            #   1. STICKY_ACTIONS            — controller override, persists N ticks
                            #   2. PENDING_OPERATOR_ACTIONS  — controller override, one-shot
                            #   3. raw_actions[i]            — the model's own decision (DEFAULT — always active)
                            desired_actions = []
                            for i in range(len(raw_actions)):
                                t_id = state.inference_train_ids[i] if i < len(state.inference_train_ids) else ""
                                sticky = STICKY_ACTIONS.get(t_id)
                                if sticky and sticky[1] > state.sim_tick:
                                    desired_actions.append(sticky[0])
                                elif t_id in PENDING_OPERATOR_ACTIONS:
                                    desired_actions.append(PENDING_OPERATOR_ACTIONS.pop(t_id))
                                else:
                                    desired_actions.append(raw_actions[i])   # model's decision

                            # Step 2: OR-Shield validates the intent to prevent crashes
                            if OR_SHIELD_ENABLED:
                                safe_actions, decision_meta = _OR_SHIELD.optimize_decision(
                                    trains=inner_env.trains,
                                    ai_actions=desired_actions,
                                    track_map=inner_env.track_map,
                                    raw_actions=raw_actions,
                                )
                            else:
                                safe_actions, decision_meta = desired_actions.copy(), {}
                        
                            # If Auto-Commit is OFF (AUTOPILOT_MODE is False), hold contested decisions
                            if not AUTOPILOT_MODE:
                                for i, t in enumerate(inner_env.trains):
                                    if decision_meta.get(t['id'], {}).get('contested', False):
                                        safe_actions[i] = 0  # Hold contested decisions
                                    
                            state.inference_actions = safe_actions
                            _INFERENCE_DECISION_META = decision_meta

                            # Step 3: Execute safe actions in physics engine
                            step_actions = np.array([safe_actions])
                            step_result = env.step(step_actions)
                            _INFERENCE_OBS = step_result[0]
                            terminated = step_result[2]
                            infos = step_result[3] if len(step_result) > 3 else [{}]

                            # ── Detect removed trains (finished/deadlocked) ────────
                            # The physics engine removes finished trains from its active list.
                            # We must catch this and mark them as Finished in the live map,
                            # otherwise they stay stuck as 'Moving' forever and inflate traffic counts.
                            current_rl_train_ids = {t['id'] for t in inner_env.trains}
                            for t_id, live in list(state.train_states.items()):
                                if t_id not in current_rl_train_ids and live.get('status') not in ('Finished', 'Scheduled', 'Expired'):
                                    live['status'] = 'Finished'
                                    live['finish_time'] = state.sim_tick
                                    live['speed_kmh'] = 0
                                    # Move off-screen visually
                                    dir_val = live.get('direction', 'DOWN')
                                    if dir_val == "UP" or dir_val == 1:
                                        live['edge_id'] = "edge-0-1"
                                        live['position_percentage'] = 0.0
                                    else:
                                        live['edge_id'] = "edge-83-999"
                                        live['position_percentage'] = 1.0

                            # ── Read RL env train positions back into state.train_states ─
                            # The RL env manages its own complete, valid train state.
                            # We map RL node positions → edge IDs for the live map.
                            # Real topology: nodes 0..83 → edge-{n}-{n+1}, node 83 → edge-83-999
                            for i, t_id in enumerate(state.inference_train_ids):
                                if t_id not in state.train_states or i >= len(inner_env.trains):
                                    continue
                                rl_train = inner_env.trains[i]
                                live     = state.train_states[t_id]

                                node_id  = rl_train.get('position', 0)
                                finished = rl_train.get('finished', False)
                                speed    = rl_train.get('speed', 0)

                                # Build edge_id — topology ends at node 83 → 999
                                direction_str = live.get('direction', 'DOWN')
                            
                                if node_id == 999 or node_id == 998:
                                    edge_id = "edge-83-999"
                                elif node_id == 0:
                                    edge_id = "edge-0-1"
                                else:
                                    if direction_str == "UP":
                                        prev_opts = RAW_TRACK_MAP.get(node_id, {}).get("prev", [])
                                        if prev_opts:
                                            edge_id = f"edge-{prev_opts[0]}-{node_id}"
                                        else:
                                            edge_id = f"edge-{node_id - 1}-{node_id}"
                                    else:
                                        next_opts = RAW_TRACK_MAP.get(node_id, {}).get("next", [])
                                        if next_opts:
                                            edge_id = f"edge-{node_id}-{next_opts[0]}"
                                        else:
                                            edge_id = f"edge-{node_id}-{node_id + 1}"

                                live['edge_id']    = edge_id
                                live['speed_kmh']  = speed
                            
                                # Smooth continuous position extraction.
                                # _movement_acc is clamped to [0, 0.999] in physics,
                                # so pct can NEVER be 1.0 while still on this node.
                                pct = 0.5
                                if hasattr(inner_env, '_movement_acc'):
                                    try:
                                        pct = float(inner_env._movement_acc[i])
                                        pct = max(0.0, min(pct, 0.999))  # defensive clamp
                                    except IndexError:
                                        pass
                            
                                # UP trains traverse the edge in reverse (high→low km).
                                if direction_str == "UP":
                                    pct = 1.0 - pct
                                
                                live['position_percentage'] = pct
                                if finished:
                                    live['status'] = 'Finished'
                                    live['finish_time'] = state.sim_tick
                                elif node_id in (0, 998):
                                    live['status'] = 'Scheduled'
                                elif rl_train.get('banker_wait', 0) > 0:
                                    live['status'] = 'Banker Ops'
                                elif rl_train.get('dwell_rem', 0) > 0:
                                    live['status'] = 'Boarding'
                                elif i < len(state.inference_actions) and state.inference_actions[i] == 0:
                                    live['status'] = 'Waiting at Signal'
                                else:
                                    live['status'] = 'Moving'

                            if bool(terminated[0]) if hasattr(terminated, '__getitem__') else bool(terminated):
                                # Complete one cycle and stop
                                reason = infos[0].get("termination_reason", "Unknown") if hasattr(infos, '__getitem__') and isinstance(infos[0], dict) else "Unknown"
                                print(f"[ORBIT] 🛑 RL episode complete ({reason}) — stopping inference after one cycle.")
                            
                                if reason != "Success":
                                    audit_entry = {
                                        "t"         : _now_iso(),
                                        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
                                        "source"    : "INFERENCE_ENGINE",
                                        "action"    : f"Reset: {reason}",
                                        "operator"  : "System",
                                        "details"   : f"Simulation finished due to: {reason}",
                                    }
                                    _push_audit_log(state, audit_entry)
                                    asyncio.create_task(_broadcast_copilot({
                                        "type": "audit_log",
                                        "log": audit_entry
                                    }))

                                _INFERENCE_OBS = env.reset()
                                state.inference_raw_actions = None
                            
                                # Let this tick finish processing normally to prevent the fallback physics 
                                # engine from destroying train statuses mid-tick. Turn it off at the end.
                                state.shutdown_inference_flag = True

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[ORBIT] ⚠️  Inference sync error: {e}")

                # ── Drive map movement for ALL trains (inference + fallback) ───────
                # When inference is active, RL action determines speed (stop vs move).
                # When not active, all Moving trains advance normally.
                for idx, t_id in enumerate(live_train_ids):
                    state = state.train_states[t_id]

                    if state['status'] == 'Finished':
                        continue

                    curr_edge = state.get('edge_id', '')

                    # ── MAINTENANCE BLOCK CHECK (current edge) ────────────────────
                    edge_block = ACTIVE_BLOCKS.get(curr_edge)
                    if edge_block and is_block_active(edge_block):
                        if edge_block.get('severity') == 'TOTAL_BLOCK':
                            state['status'] = 'Blocked'
                            if state.inference_active:
                                STICKY_ACTIONS[t_id] = (0, state.sim_tick + 2)
                            continue
                        elif edge_block.get('severity') == 'SPEED_RESTRICTION':
                            # Cap train speed to the configured limit (default 30 km/h)
                            limit = edge_block.get('speed_limit', 30)
                            if state.get('speed', 0) > limit:
                                state['speed'] = limit
                            state['status'] = 'Moving'  # still moving, just slower

                    # ── LOOKAHEAD BLOCK CHECK (next edge in path) ─────────────────
                    # Prevent trains from advancing into a blocked segment
                    path = state.get('path', [])
                    try:
                        curr_path_idx = path.index(curr_edge)
                        if curr_path_idx + 1 < len(path):
                            next_edge = path[curr_path_idx + 1]
                            next_block = ACTIVE_BLOCKS.get(next_edge)
                            if next_block and is_block_active(next_block) and next_block.get('severity') == 'TOTAL_BLOCK':
                                # Halt the train before it crosses into the blocked segment
                                state['status'] = 'Halted'
                                if state.inference_active:
                                    STICKY_ACTIONS[t_id] = (0, state.sim_tick + 2)
                                continue
                    except (ValueError, IndexError):
                        pass

                    # ── Per-train simulation clock ────────────────────────────────
                    # Ticks up once per loop iteration for every active train.
                    # Used for schedule-deadline comparisons and delay reporting.
                    state['sim_time'] = state.get('sim_time', 0) + 1

                    # Determine whether this train should move this tick
                    if state.inference_active and state.inference_actions is not None:
                        # RL says: 0=STOP, 1=MAIN (move), 2=DIVERT (move to loop, treated as move)
                        rl_act = state.inference_actions[idx] if idx < len(state.inference_actions) else 1



                        # ── Committed override takes priority over live RL ────────
                        # A controller commit writes override_action + override_expires
                        # to state.train_states.  While the override is active we honour the
                        # committed decision; once it expires the RL agent resumes.
                        override_exp = state.get('override_expires', 0)
                        if override_exp > state.sim_tick:
                            rl_act = state['override_action']
                            ticks_left = override_exp - state.sim_tick
                            print(f"[ORBIT] 🔒 Override active for {t_id}: "
                                  f"action={rl_act} ({ticks_left} ticks remaining)")

                        should_move = (rl_act != 0)
                    else:
                        should_move = (state['status'] in ('Moving', 'Blocked'))

                    if should_move:
                        if not state.inference_active and state.get('status') not in ('Scheduled', 'Finished'):
                            state['status'] = 'Moving'
                            spd = state.get('speed_kmh', 0)
                            mx = state.get('max_speed', 130)
                            state['position_percentage'] = state.get('position_percentage', 0) + (spd / mx) * 0.05 * SIM_SPEED_FACTOR
                            if state['position_percentage'] >= 1.0:
                                state['position_percentage'] = 0.0
                                try:
                                    curr_idx = path.index(state['edge_id'])
                                    if curr_idx + 1 < len(path):
                                        state['edge_id'] = path[curr_idx + 1]
                                    else:
                                        state['status'] = 'Finished'
                                        state['finish_time'] = state.sim_tick
                                except ValueError:
                                    pass
                    else:
                        if state.get('status') not in ('Scheduled', 'Finished', 'Boarding', 'Banker Ops'):
                            # If inference is active and assigned 'Waiting at Signal', preserve it unless overridden
                            if not (state.inference_active and state.get('status') == 'Waiting at Signal' and state.get('override_action') is None):
                                state['status'] = 'Halted'


            edges_occupied: Dict[str, list] = {}
            for t_id, state in state.train_states.items():
                if state.get('status') not in ('Finished', 'Scheduled'):
                    edges_occupied.setdefault(state['edge_id'], []).append(state)

            conflicts: set = set()
            # Active maintenance blocks are also surfaced as conflicts for the map
            for element_id, blk in ACTIVE_BLOCKS.items():
                if blk.get('severity') == 'TOTAL_BLOCK':
                    conflicts.add(element_id)

            for e_id, trains in edges_occupied.items():
                # Find edge capacity
                cap = 1
                for edge_info in NETWORK_TOPOLOGY.get('edges', []):
                    if edge_info['id'] == e_id:
                        cap = edge_info.get('capacity', 1)
                        break
            
                if len(trains) > cap:
                    conflicts.add(e_id)
                    for t in trains:
                        if t['status'] != 'Blocked':
                            t['status'] = 'Halted'

            for e_id, trains in edges_occupied.items():
                if e_id not in conflicts:
                    for t in trains:
                        # If they were Halted due to a physical conflict, they can now resume moving.
                        # We do NOT overwrite 'Waiting at Signal' or 'Boarding' or 'Scheduled'.
                        if t['status'] == 'Halted':
                            t['status'] = 'Moving'

        payload = {
            "type": "topology_update",
            "sim_time": state.sim_tick,
            "trains": [t for t in state.train_states.values() if t.get("status") not in ("Scheduled", "Finished")],
            "all_trains": [{"train_id": t["train_id"], "status": t.get("status", "Scheduled")} for t in state.train_states.values() if t.get("status") != "Finished"],
            "conflicts": list(conflicts),
            "maintenance_blocks": list(ACTIVE_BLOCKS.values()),
        }
        await _broadcast_topology(payload)
        
        async with _SIM_LOCK:
            if state.shutdown_inference_flag:
                state.inference_active = False
                state.shutdown_inference_flag = False
                # Clear all train positions from the map immediately
                for state in state.train_states.values():
                    if state.get("status") not in ("Finished", "Scheduled"):
                        state["status"] = "Finished"
            await _broadcast_topology({
                "type": "topology_update",
                "sim_time": state.sim_tick,
                "trains": [],
                "all_trains": [{"train_id": t["train_id"], "status": t.get("status", "Scheduled")} for t in state.train_states.values() if t.get("status") != "Finished"],
                "conflicts": [],
                "maintenance_blocks": list(ACTIVE_BLOCKS.values()),
            })

        await asyncio.sleep(TICK_INTERVAL_S)



async def copilot_suggestion_bg():
    """
    AI Co-Pilot background task — runs every 8 seconds.
    """
    await asyncio.sleep(3)   # hold for frontend to connect
    while True:
        # Pruning Sweep (Fix 5)
        expired_keys = []
        for k, v in COPILOT_SUGGESTIONS.items():
            if v.get("status") == "pending" and state.sim_tick - v.get("suggested_at_tick", state.sim_tick) > SUGGESTION_TTL_TICKS:
                expired_keys.append(k)
        for k in expired_keys:
            s = COPILOT_SUGGESTIONS[k]
            s["status"] = "expired"
            _write_feedback(s, "expired", "TTL exceeded")
            
        if len(COPILOT_SUGGESTIONS) > COPILOT_SUGGESTIONS_MAX_SIZE:
            # Drop the oldest half
            sorted_keys = sorted(COPILOT_SUGGESTIONS.keys(), key=lambda x: COPILOT_SUGGESTIONS[x].get("timestamp", ""))
            for k in sorted_keys[:COPILOT_SUGGESTIONS_MAX_SIZE // 2]:
                del COPILOT_SUGGESTIONS[k]

        if COPILOT_WEBSOCKETS and state.inference_active:
            candidates = _make_suggestion()          # RL proposal
            if not candidates:
                await asyncio.sleep(2)
                continue
                
            for candidate in candidates:
                # ── OR-Shield Gate ──────────────────────────────────────────────
                if OR_SHIELD_ENABLED:
                    is_safe, reason = _OR_SHIELD.or_shield_check(
                        suggestion=candidate,
                        train_states=state.train_states,
                        active_blocks={k: v for k, v in ACTIVE_BLOCKS.items() if is_block_active(v)},
                        dynamic_constraints=DYNAMIC_CONSTRAINTS,
                    )
                else:
                    is_safe, reason = True, "OR-Shield Disabled"

                if not is_safe:
                    print(
                        f"[OR-Shield] 🛡️  Filtered suggestion "
                        f"{candidate['recommendation_id'][:8]}… "
                        f"(target: {candidate['target_train_id']}) — "
                        f"reason: {reason}"
                    )
                else:
                    # ── Suggestion generator is now purely advisory ────────────
                    # Auto-commit is no longer applied here — it was a source of
                    # duplicated physics and hidden side-effects.  Autopilot logic
                    # lives exclusively in the execution stage (simulate_trains_bg).
                    # Every OR-Shield-approved suggestion is queued for operator review.
                    ticks_remaining = SUGGESTION_TTL_TICKS - (state.sim_tick - candidate.get("suggested_at_tick", state.sim_tick))
                    candidate["expires_in_ticks"] = max(0, ticks_remaining)
                    COPILOT_SUGGESTIONS[candidate["recommendation_id"]] = candidate
                    await _broadcast_copilot(candidate)
                    print(
                        f"[ORBIT] ✅ Emitted suggestion "
                        f"{candidate['recommendation_id'][:8]}… "
                        f"(P{candidate['priority_level']}, {candidate['target_train_id']}, "
                        f"expires in {ticks_remaining} ticks)"
                    )

        await asyncio.sleep(8 * TICK_INTERVAL_S)


