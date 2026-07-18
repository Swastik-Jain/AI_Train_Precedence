from typing import Dict, Any, Tuple
import os

from state import SimulationState
from config import DOWN_PATH, UP_PATH

def _get_sim_brain(state: SimulationState):
    if state.sim_model is not None:
        return state.sim_model, state.sim_env

    # ── Level 6 checkpoint (Latest best) ──────────────────
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(
        base_dir, "ai", "models", "Phase3", "L6_25Trains_Best_v5", "best_model.zip"
    )
    stats_path = os.path.join(
        base_dir, "ai", "models", "Phase3", "vec_normalize_L6_25Trains.pkl"
    )

    if not os.path.exists(model_path):
        print(f"[SIM-BRAIN] ⚠️  Model not found at {model_path} — falling back to OR-Tools only.")
        return None, None

    try:
        os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import sys
        sys.path.append(base_dir) # to allow importing train_env
        from train_env import TrainDispatchEnv

        def make_env():
            e = TrainDispatchEnv()
            e.set_difficulty(25)
            e.inference_mode = True
            return e

        raw_env = DummyVecEnv([make_env])

        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, raw_env)
            env.training = False
            env.norm_reward = False
            print("[SIM-BRAIN] 📊 VecNormalize stats loaded for 15-train model.")
        else:
            env = raw_env
            print("[SIM-BRAIN] ⚠️  No VecNormalize stats found — running without normalization.")

        model = MaskablePPO.load(
            model_path,
            env=None,
            device="cpu"
        )
        state.sim_model = model
        state.sim_env   = env
        print("✅ [SIM-BRAIN] 25-Train MaskablePPO model loaded for sandbox analysis.")
        return model, env
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[SIM-BRAIN] ❌ Failed to load model: {exc}")
        return None, None

def start_inference(state: SimulationState) -> Dict[str, Any]:
    state.inference_active = False
    state.inference_obs = None
    state.inference_actions = None
    state.inference_raw_actions = None
    if not state.last_or_schedule:
        return {
            "status" : "error",
            "message": "No schedule found. Please generate a conflict-free schedule on the Fleet Status page first."
        }

    model, env = _get_sim_brain(state)
    if not (model and env):
        return {"status": "error", "message": "RL model could not be loaded."}

    new_states: Dict[str, Any] = {}
    ordered_ids = []

    for t_id, cfg in state.fleet_registry.items():
        path = cfg.get("path", [])
        if not path:
            direction_str = "DOWN" if cfg.get("direction", "DOWN") in (1, "DOWN") else "UP"
            path = DOWN_PATH if direction_str == "DOWN" else UP_PATH
            cfg["path"] = path

        cfg["id"] = t_id
        ordered_ids.append(t_id)
        new_states[t_id] = {
            "train_id"             : t_id,
            "train_type"           : cfg.get("train_type", "Express"),
            "edge_id"              : path[0],
            "position_percentage"  : 0.0,
            "status"               : "Scheduled",
            "speed_kmh"            : cfg.get("max_speed", 90),
            "path"                 : path,
            "priority"             : cfg.get("priority", 6),
            "direction"            : cfg.get("direction", "DOWN"),
            "scheduled_departure"  : cfg.get("start_time", 0),
            "scheduled_arrival"    : cfg.get("deadline", 120),
            "delay_mins"           : 0,
            "sim_time"             : 0,
            "override_action"      : 1,
            "override_expires"     : 0,
        }

    if new_states:
        state.train_states = new_states
        state.inference_train_ids = ordered_ids

    inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]

    formatted_schedule = {}
    for t_id, t_sched in state.last_or_schedule.items():
        cfg = state.fleet_registry.get(t_id, {})
        formatted_schedule[t_id] = {
            'start_time': cfg.get("start_time", 0),
            'deadline': cfg.get("deadline", 120),
            'direction': "DOWN" if cfg.get("direction", 1) in (1, "DOWN") else "UP",
            'stops': list(t_sched.keys())
        }
    
    inner_env.set_custom_schedule(
        fleet=list(state.fleet_registry.values()),
        schedule=formatted_schedule
    )
    state.inference_obs     = env.reset()
    from services.maintenance_service import sync_blocks_to_rl_env
    sync_blocks_to_rl_env(state)
    state.inference_obs     = np.array(env.env_method("_get_observation"))
    state.inference_actions = None
    state.sim_tick           = 0
    state.inference_sim_time = 0   # reset episodic clock alongside RL env
    state.inference_active   = True

    print(f"[ORBIT] 🚀 Inference started. {len(state.train_states)} trains seeded from OR schedule.")
    return {
        "status" : "started",
        "active" : True,
        "trains" : len(state.train_states),
    }

def stop_inference(state: SimulationState) -> Dict[str, Any]:
    state.inference_active = False
    state.train_states.clear()
    state.inference_train_ids.clear()
    return {"status": "stopped", "active": False}

import asyncio
import time
import uuid
import numpy as np
from datetime import datetime, timezone
import config
from services.maintenance_service import is_block_active
from services import system_service
async def simulate_trains_bg(state, broadcast_topology, broadcast_copilot, _sync_blocks_to_rl_env, _push_audit_log):
    

    state.train_states = {}
    _spawned = False

    # Real mainline corridor paths — defined at module level, referenced here.
    # DOWN: CSMT → MANMAD (nodes 0→83→999)  UP: MANMAD → CSMT (reverse)

    while True:
        if not state.inference_active:
            # Broadcast a clear payload so the frontend doesn't show stale/ghost trains
            await broadcast_topology({
                "type": "topology_update",
                "sim_time": state.sim_tick,
                "trains": [t for t in state.train_states.values()
                           if t.get("status") not in ("Scheduled", "Finished")],
                "all_trains": [{"train_id": t["train_id"], "status": t.get("status", "Scheduled")} for t in state.train_states.values() if t.get("status") != "Finished"],
                "conflicts": [],
                "maintenance_blocks": list(state.active_blocks.values()),
                "ghat_queue": {"ksr": {"count": 0, "train_ids": []}, "igp": {"count": 0, "train_ids": []}},
            })
            await asyncio.sleep(1.0)
            continue
            
        async with state.sim_lock:
            state.sim_tick += 1
            now_ts = time.time()
            # TTL Pruning for state.dynamic_constraints
            expired = [c_id for c_id, c in list(state.dynamic_constraints.items()) if c.get('expires_at', float('inf')) < now_ts]
            for c_id in expired:
                del state.dynamic_constraints[c_id]
                _push_audit_log(state, {
                    "t": system_service._now_iso(),
                    "source": "SIMULATION_ENGINE",
                    "action": f"Constraint {c_id} automatically expired",
                    "operator": "SYSTEM",
                    "status": "Expired",
                    "statusType": "info"
                })
            
            # Prune Finished trains to prevent unbounded state growth
            # (Disabled so that dashboard_service can continue to count them for punctuality)
            # finished_trains = [t_id for t_id, live in state.train_states.items() if live.get("status") == "Finished"]
            # for t_id in finished_trains:
            #     state.train_states.pop(t_id, None)
            #     state.fleet_registry.pop(t_id, None)
            
            # Prune expired maintenance blocks
            expired_blocks = []
            now_dt = datetime.now(timezone.utc)
            for b_id, block in list(state.active_blocks.items()):
                try:
                    end = datetime.fromisoformat(block.get("end_time", "").replace("Z", "+00:00"))
                    if now_dt > end:
                        expired_blocks.append(b_id)
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"[WARN] Failed to parse end_time for maintenance block {b_id}: {e}")
                    pass # If it fails to parse, leave it until manually removed
                
            for b_id in expired_blocks:
                blk = state.active_blocks.pop(b_id)
                asyncio.create_task(broadcast_topology({
                    "type": "MAINTENANCE_CLEARED",
                    "element_id": b_id,
                }))
                _push_audit_log(state, {
                    "t": system_service._now_iso(),
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
                _sync_blocks_to_rl_env(state)

            if state.system_lockdown:
                for t_id, t_state in state.train_states.items():
                    t_state['status'] = 'Halted'
            else:
                # ── Build the ordered list of active trains once per tick ──────────
                live_train_ids = list(state.train_states.keys())

                if state.inference_active:
                    try:
                        model, env = _get_sim_brain(state)
                        if model and env:
                            inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]

                            # ── Get observation and predict ────────────────────────
                            if state.inference_obs is None:
                                state.inference_obs = env.reset()
                                _sync_blocks_to_rl_env(state)
                                state.inference_obs = np.array(env.env_method("_get_observation"))

                            action_masks = np.array(env.env_method("get_action_mask"))
                            action, _ = model.predict(
                                state.inference_obs, deterministic=True, action_masks=action_masks
                            )

                            import torch
                            obs_tensor = model.policy.obs_to_tensor(state.inference_obs)[0]
                            act_list = list(action[0]) if hasattr(action[0], '__iter__') else list(action)
                            try:
                                with torch.no_grad():
                                    dist = model.policy.get_distribution(obs_tensor)
                                    probs_list = []
                                    if hasattr(dist, "distributions") and isinstance(dist.distributions, list):
                                        # MultiDiscrete case (newer SB3 / sb3-contrib)
                                        for d, a in zip(dist.distributions, act_list):
                                            p = torch.exp(d.log_prob(torch.tensor(a).to(model.device))).item()
                                            probs_list.append(p)
                                    elif hasattr(dist, "distribution") and isinstance(dist.distribution, list):
                                        # MultiDiscrete case (older SB3)
                                        for d, a in zip(dist.distribution, act_list):
                                            p = torch.exp(d.log_prob(torch.tensor(a).to(model.device))).item()
                                            probs_list.append(p)
                                    else:
                                        # Discrete or fallback
                                        action_tensor = torch.tensor(act_list).to(model.device)
                                        log_probs = dist.log_prob(action_tensor)
                                        p_arr = torch.exp(log_probs).cpu().numpy()
                                        if p_arr.ndim == 0:
                                            probs_list = [p_arr.item()] * len(act_list)
                                        else:
                                            probs_list = list(p_arr)
                                state.inference_action_probs = probs_list
                            except Exception as e:
                                print(f"[ORBIT] ⚠️  Inference probability extraction error: {e}")
                                state.inference_action_probs = [0.85] * len(act_list)


                            raw_actions = list(action[0]) if hasattr(action[0], '__iter__') else list(action)
                            state.inference_raw_actions = raw_actions

                            # ── Advisory stage (read-only) ────────────────────────
                            # Store the model's full proposal without applying it.
                            # This is the source of truth for the transparency endpoint
                            # and for state.autopilot_mode — it never directly touches state.train_states.
                            for _i, _t_id in enumerate(state.inference_train_ids):
                                if _i < len(raw_actions):
                                    state.latest_model_proposal[_t_id] = int(raw_actions[_i])

                            # ── Execution stage: autonomous-by-default, override-on-top ─────
                            #   1. state.sticky_actions            — controller override, persists N ticks
                            #   2. state.pending_operator_actions  — controller override, one-shot
                            #   3. raw_actions[i]            — the model's own decision (DEFAULT — always active)
                            desired_actions = []
                            for i in range(len(raw_actions)):
                                t_id = state.inference_train_ids[i] if i < len(state.inference_train_ids) else ""
                                sticky = state.sticky_actions.get(t_id)
                                if sticky and sticky[1] > state.sim_tick:
                                    desired_actions.append(sticky[0])
                                elif t_id in state.pending_operator_actions:
                                    desired_actions.append(state.pending_operator_actions.pop(t_id))
                                else:
                                    desired_actions.append(raw_actions[i])   # model's decision

                            # Step 2: OR-Shield validates the intent to prevent crashes
                            if state.or_shield_enabled:
                                node_km = {nid: info.get('km', 0.0) for nid, info in inner_env.track_map.items()}
                                safe_actions, decision_meta = state.or_shield.optimize_decision(
                                    trains=inner_env.trains,
                                    ai_actions=desired_actions,
                                    track_map=inner_env.track_map,
                                    node_km=node_km,
                                    raw_actions=raw_actions,
                                )
                            else:
                                safe_actions, decision_meta = desired_actions.copy(), {}
                        
                            # If Auto-Commit is OFF (state.autopilot_mode is False), hold contested decisions
                            if not state.autopilot_mode:
                                for i, t in enumerate(inner_env.trains):
                                    if decision_meta.get(t['id'], {}).get('contested', False):
                                        safe_actions[i] = 0  # Hold contested decisions
                                    
                            state.inference_actions = safe_actions
                            state.inference_decision_meta = decision_meta

                            # Step 3: Execute safe actions in physics engine
                            step_actions = np.array([safe_actions])
                            step_result = env.step(step_actions)
                            state.inference_obs = step_result[0]
                            terminated = step_result[2]
                            infos = step_result[3] if len(step_result) > 3 else [{}]
                            # ── Capture RL env episodic sim_time (resets each episode) ──
                            # state.sim_tick is a global wall-clock that NEVER resets.
                            # Deadlines in fleet_registry are relative to episode start (0).
                            # We MUST use the RL env's own sim_time for correct comparisons.
                            state.inference_sim_time = getattr(inner_env, 'sim_time', state.sim_tick)

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
                                committed_next = rl_train.get('committed_next_node')

                                if not committed_next:
                                    # Fallbacks if RL engine hasn't chosen next node yet
                                    edge_id = live.get('path', ['edge-0-1'])[0]
                                elif node_id == 999 or node_id == 998:
                                    edge_id = "edge-83-999"
                                elif node_id == 0:
                                    edge_id = "edge-0-1"
                                else:
                                    # Use the RL env's own committed next-node — this is the
                                    # real MAIN/DIVERT decision train_env.py made for this
                                    # train this step, not a re-derived guess.
                                    committed_next = rl_train.get('committed_next_node')
                                    if committed_next is None:
                                        committed_next = node_id  # end of line / no data — draw a self-loop-ish fallback rather than crash

                                    if direction_str == "UP":
                                        edge_id = f"edge-{committed_next}-{node_id}"
                                    else:
                                        edge_id = f"edge-{node_id}-{committed_next}"

                                live['edge_id']    = edge_id
                                live['position_node'] = node_id
                                live['speed_kmh']  = speed
                            
                                # Smooth continuous position extraction.
                                # _movement_acc is physical distance in km. We must divide by edge length
                                # to get a true percentage [0.0, 1.0) for the frontend interpolation.
                                pct = 0.5
                                if hasattr(inner_env, '_movement_acc'):
                                    try:
                                        acc_val = float(inner_env._movement_acc[i])
                                        
                                        # Use the RL env's own committed next-node (same field the
                                        # edge_id resolution above reads) so the progress percentage
                                        # is measured along the edge actually being displayed,
                                        # not a re-guessed one.
                                        target_node = rl_train.get('committed_next_node')
                                        if target_node is None:
                                            target_node = node_id - 1 if direction_str == "UP" else node_id + 1

                                        km1 = inner_env.get_node_km(node_id) if hasattr(inner_env, 'get_node_km') else 0
                                        km2 = inner_env.get_node_km(target_node) if hasattr(inner_env, 'get_node_km') else 1
                                        
                                        node_st = inner_env.track_map.get(node_id, {}).get('station')
                                        tgt_st = inner_env.track_map.get(target_node, {}).get('station')
                                        if node_st and tgt_st and node_st == tgt_st:
                                            dist_to_next = max(1.0, abs(km2 - km1))
                                        else:
                                            dist_to_next = max(0.1, abs(km2 - km1))
                                            
                                        pct = acc_val / dist_to_next
                                        pct = max(0.0, min(pct, 0.999))  # defensive clamp
                                    except (TypeError, ValueError, KeyError, AttributeError) as e:
                                        print(f"[WARN] position_percentage calc failed for {t_id}: {e}")
                            
                                # UP trains traverse the edge in reverse (high→low km).
                                if direction_str == "UP":
                                    pct = 1.0 - pct
                                
                                live['position_percentage'] = pct
                                if finished:
                                    live['status'] = 'Finished'
                                    if 'finish_step' not in live or live['finish_step'] is None:
                                        live['finish_step'] = rl_train.get('finish_step')
                                    if 'finish_time' not in live:
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
                                        "t"         : system_service._now_iso(),
                                        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
                                        "source"    : "INFERENCE_ENGINE",
                                        "action"    : f"Reset: {reason}",
                                        "operator"  : "System",
                                        "details"   : f"Simulation finished due to: {reason}",
                                    }
                                    _push_audit_log(state, audit_entry)
                                    asyncio.create_task(broadcast_copilot({
                                        "type": "audit_log",
                                        "log": audit_entry
                                    }))

                                state.inference_obs = env.reset()
                                _sync_blocks_to_rl_env(state)
                                state.inference_obs = np.array(env.env_method("_get_observation"))
                                state.inference_raw_actions = None
                            
                                # Let this tick finish processing normally to prevent the fallback physics 
                                # engine from destroying train statuses mid-tick. Turn it off at the end.
                                state.shutdown_inference_flag = True
                            
                            state.consecutive_inference_errors = 0

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[ORBIT] ⚠️  Inference sync error: {e}")

                        state.consecutive_inference_errors += 1
                        _push_audit_log(state, {
                            "t": system_service._now_iso(),
                            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                            "source": "INFERENCE_ENGINE",
                            "action": f"Inference tick failed ({state.consecutive_inference_errors} consecutive): {e}",
                            "operator": "SYSTEM",
                            "status": "Error",
                            "statusType": "error",
                        })

                        if state.consecutive_inference_errors >= 5:
                            state.inference_active = False
                            state.consecutive_inference_errors = 0
                            _push_audit_log(state, {
                                "t": system_service._now_iso(),
                                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                                "source": "INFERENCE_ENGINE",
                                "action": "Inference auto-stopped after 5 consecutive tick failures — check server logs.",
                                "operator": "SYSTEM",
                                "status": "Halted",
                                "statusType": "error",
                            })

                # ── Drive map movement for ALL trains (inference + fallback) ───────
                # When inference is active, RL action determines speed (stop vs move).
                # When not active, all Moving trains advance normally.
                for idx, t_id in enumerate(live_train_ids):
                    t_state = state.train_states[t_id]

                    if t_state['status'] == 'Finished':
                        continue

                    curr_edge = t_state.get('edge_id', '')

                    # ── MAINTENANCE BLOCK CHECK (current edge) ────────────────────
                    edge_block = state.active_blocks.get(curr_edge)
                    if edge_block and is_block_active(edge_block):
                        if edge_block.get('severity') == 'TOTAL_BLOCK':
                            t_state['status'] = 'Blocked'
                            if state.inference_active:
                                state.sticky_actions[t_id] = (0, state.sim_tick + 2)
                            continue
                        elif edge_block.get('severity') == 'SPEED_RESTRICTION':
                            # Cap train speed to the configured limit (default 30 km/h)
                            limit = edge_block.get('speed_limit', 30)
                            if t_state.get('speed', 0) > limit:
                                t_state['speed'] = limit
                            t_state['status'] = 'Moving'  # still moving, just slower

                    # ── LOOKAHEAD BLOCK CHECK (next edge in path) ─────────────────
                    # Prevent trains from advancing into a blocked segment
                    path = t_state.get('path', [])
                    try:
                        curr_path_idx = path.index(curr_edge)
                        if curr_path_idx + 1 < len(path):
                            next_edge = path[curr_path_idx + 1]
                            next_block = state.active_blocks.get(next_edge)
                            if next_block and is_block_active(next_block) and next_block.get('severity') == 'TOTAL_BLOCK':
                                # Halt the train before it crosses into the blocked segment
                                t_state['status'] = 'Halted'
                                if state.inference_active:
                                    state.sticky_actions[t_id] = (0, state.sim_tick + 2)
                                continue
                    except ValueError:
                        # curr_edge isn't in this train's static reference path — expected
                        # whenever the train is on a loop/divert edge not covered by the
                        # DOWN_PATH/UP_PATH reference list. Not an error.
                        pass

                    # ── Per-train simulation clock ────────────────────────────────
                    # Ticks up once per loop iteration for every active train.
                    # Used for schedule-deadline comparisons and delay reporting.
                    t_state['sim_time'] = t_state.get('sim_time', 0) + 1

                    # Determine whether this train should move this tick
                    if state.inference_active and state.inference_actions is not None:
                        # RL says: 0=STOP, 1=MAIN (move), 2=DIVERT (move to loop, treated as move)
                        rl_act = state.inference_actions[idx] if idx < len(state.inference_actions) else 1



                        # ── Committed override takes priority over live RL ────────
                        # A controller commit writes override_action + override_expires
                        # to state.train_states.  While the override is active we honour the
                        # committed decision; once it expires the RL agent resumes.
                        override_exp = t_state.get('override_expires', 0)
                        if override_exp > state.sim_tick:
                            rl_act = t_state['override_action']
                            ticks_left = override_exp - state.sim_tick
                            print(f"[ORBIT] 🔒 Override active for {t_id}: "
                                  f"action={rl_act} ({ticks_left} ticks remaining)")

                        should_move = (rl_act != 0)
                    else:
                        should_move = (t_state['status'] in ('Moving', 'Blocked'))

                    if should_move:
                        if not state.inference_active and t_state.get('status') not in ('Scheduled', 'Finished'):
                            t_state['status'] = 'Moving'
                            spd = t_state.get('speed_kmh', 0)
                            mx = t_state.get('max_speed', 130)
                            t_state['position_percentage'] = t_state.get('position_percentage', 0) + (spd / mx) * 0.05 * config.SIM_SPEED_FACTOR
                            if t_state['position_percentage'] >= 1.0:
                                t_state['position_percentage'] = 0.0
                                try:
                                    curr_idx = path.index(t_state['edge_id'])
                                    if curr_idx + 1 < len(path):
                                        t_state['edge_id'] = path[curr_idx + 1]
                                    else:
                                        t_state['status'] = 'Finished'
                                        t_state['finish_time'] = state.sim_tick
                                except ValueError:
                                    pass
                    else:
                        if t_state.get('status') not in ('Scheduled', 'Finished', 'Boarding', 'Banker Ops'):
                            # If inference is active and assigned 'Waiting at Signal', preserve it unless overridden
                            if not (state.inference_active and t_state.get('status') == 'Waiting at Signal' and t_state.get('override_action') is None):
                                t_state['status'] = 'Halted'

            if state.sim_tick % 100 == 0:
                print(f"[HEARTBEAT] Sim tick: {state.sim_tick}")

            # Collect trains by their real physical position (node_id)
            nodes_occupied: Dict[int, list] = {}
            for t_id, t_state in state.train_states.items():
                if t_state.get('status') not in ('Finished', 'Scheduled'):
                    real_node = t_state.get('position_node', t_state.get('position', 0))
                    nodes_occupied.setdefault(real_node, []).append(t_state)

            occupied_by_node = {
                node_id: trains[0]
                for node_id, trains in nodes_occupied.items()
            }
            ghat_queue = {"ksr": {"count": 0, "train_ids": []}, "igp": {"count": 0, "train_ids": []}}
            try:
                _, env = _get_sim_brain(state)
                if env:
                    inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
                    ksr_ids = inner_env.ghat_token.compute_queue(inner_env.track_map, occupied_by_node, 'KSR')
                    igp_ids = inner_env.ghat_token.compute_queue(inner_env.track_map, occupied_by_node, 'IGP')
                    ghat_queue = {
                        "ksr": {"count": len(ksr_ids), "train_ids": ksr_ids},
                        "igp": {"count": len(igp_ids), "train_ids": igp_ids},
                    }
            except Exception as e:
                print(f"[WARN] ghat_queue computation failed: {e}")

            conflicts: set = set()
            train_conflicts: set = set()
            
            # Active maintenance blocks are also surfaced as conflicts for the map
            for element_id, blk in state.active_blocks.items():
                if blk.get('severity') == 'TOTAL_BLOCK':
                    conflicts.add(element_id)

            # Node-based train-crowding check using identical logic to physics engine
            try:
                _, env = _get_sim_brain(state)
                inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
            except Exception:
                inner_env = None

            if inner_env:
                for node_id, trains in nodes_occupied.items():
                    node_cap = inner_env.track_map.get(node_id, {}).get('capacity', 1)
                    node_occ = inner_env.get_node_occupancy(node_id)
                
                    # Using the identical condition that train_env.py uses to block moves
                    if len(trains) > node_cap:
                        for t in trains:
                            train_conflicts.add(t['train_id'])
                            if t['status'] != 'Blocked':
                                t['status'] = 'Halted'

            # Resume trains that are no longer over physical capacity
            for t_id, t_state in state.train_states.items():
                if t_id not in train_conflicts:
                    # If they were Halted due to a physical conflict, they can now resume moving.
                    # We do NOT overwrite 'Waiting at Signal' or 'Boarding' or 'Scheduled'.
                    if t_state.get('status') == 'Halted':
                        t_state['status'] = 'Moving'

            if hasattr(state, '_active_conflicts'):
                state._active_conflicts.intersection_update(conflicts)

        payload = {
            "type": "topology_update",
            "sim_time": state.sim_tick,
            "tick_interval_s": state.tick_interval_s,
            "trains": [t for t in state.train_states.values() if t.get("status") not in ("Scheduled", "Finished")],
            "all_trains": [{"train_id": t["train_id"], "status": t.get("status", "Scheduled")} for t in state.train_states.values() if t.get("status") != "Finished"],
            "conflicts": list(conflicts),
            "train_conflicts": list(train_conflicts),
            "maintenance_blocks": list(state.active_blocks.values()),
            "token_trains": [],
            "ghat_queue": ghat_queue,
        }
        
        if state.inference_active:
            try:
                model, env = _get_sim_brain(state)
                if env:
                    inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
                    payload["token_trains"] = inner_env.ghat_token.status().get("trains_in_block", [])
            except Exception as e:
                print(f"[WARN] Failed to read ghat token status for broadcast: {e}")

        await broadcast_topology(payload)
        
        async with state.sim_lock:
            if state.shutdown_inference_flag:
                state.inference_active = False
                state.shutdown_inference_flag = False
                # Clear all train positions from the map immediately
                for t_state in state.train_states.values():
                    if t_state.get("status") not in ("Finished", "Scheduled"):
                        t_state["status"] = "Finished"
                await broadcast_topology({
                    "type": "topology_update",
                    "sim_time": state.sim_tick,
                    "trains": [],
                    "all_trains": [{"train_id": t["train_id"], "status": t.get("status", "Scheduled")} for t in state.train_states.values() if t.get("status") != "Finished"],
                    "conflicts": [],
                    "train_conflicts": [],
                    "maintenance_blocks": list(state.active_blocks.values()),
                    "ghat_queue": ghat_queue,
                })

        await asyncio.sleep(state.tick_interval_s)



async def copilot_suggestion_bg(state, broadcast_copilot, _write_feedback, _make_suggestion):
    """
    AI Co-Pilot background task — runs every 8 seconds.
    """
    await asyncio.sleep(3)   # hold for frontend to connect
    while True:
        # Pruning Sweep (Fix 5)
        expired_keys = []
        for k, v in state.copilot_suggestions.items():
            if v.get("status") == "pending" and state.sim_tick - v.get("suggested_at_tick", state.sim_tick) > config.SUGGESTION_TTL_TICKS:
                expired_keys.append(k)
        for k in expired_keys:
            s = state.copilot_suggestions[k]
            s["status"] = "expired"
            _write_feedback(s, "expired", "TTL exceeded")
            
        if len(state.copilot_suggestions) > config.COPILOT_SUGGESTIONS_MAX_SIZE:
            # Drop the oldest half
            sorted_keys = sorted(state.copilot_suggestions.keys(), key=lambda x: state.copilot_suggestions[x].get("timestamp", ""))
            for k in sorted_keys[:config.COPILOT_SUGGESTIONS_MAX_SIZE // 2]:
                del state.copilot_suggestions[k]

        if state.copilot_websockets and state.inference_active:
            candidates = _make_suggestion(state)          # RL proposal
            if not candidates:
                await asyncio.sleep(2)
                continue
                
            for candidate in candidates:
                # ── OR-Shield Gate ──────────────────────────────────────────────
                if state.or_shield_enabled:
                    is_safe, reason = state.or_shield.or_shield_check(
                        suggestion=candidate,
                        train_states=state.train_states,
                        active_blocks={k: v for k, v in state.active_blocks.items() if is_block_active(v)},
                        dynamic_constraints=state.dynamic_constraints,
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
                    ticks_remaining = config.SUGGESTION_TTL_TICKS - (state.sim_tick - candidate.get("suggested_at_tick", state.sim_tick))
                    candidate["expires_in_ticks"] = max(0, ticks_remaining)
                    state.copilot_suggestions[candidate["recommendation_id"]] = candidate
                    await broadcast_copilot(candidate)
                    print(
                        f"[ORBIT] ✅ Emitted suggestion "
                        f"{candidate['recommendation_id'][:8]}… "
                        f"(P{candidate['priority_level']}, {candidate['target_train_id']}, "
                        f"expires in {ticks_remaining} ticks)"
                    )

        await asyncio.sleep(8 * state.tick_interval_s)



