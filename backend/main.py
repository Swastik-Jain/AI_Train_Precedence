from fastapi import FastAPI, Depends, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Tuple, List, Literal
from pydantic import BaseModel
import asyncio
import json
import random
import uuid
import time
import os
import numpy as np
from datetime import datetime, timezone

import database
import schema
import crud
from database import TrainPosition
from ai.config import ACTIVE_FLEET as TRAIN_CONFIG
from or_tools.smart_optimizer import SmartOptimizer
from topology import get_network_topology


database.init_db()

app = FastAPI(title="ORBIT: Operational Rail Backbone")

# ---------------------------------------------------------------------------
# Topology Simulation State
# ---------------------------------------------------------------------------
TOPOLOGY_DATA = get_network_topology()
NETWORK_TOPOLOGY = {"nodes": TOPOLOGY_DATA["nodes"], "edges": TOPOLOGY_DATA["edges"]}
RAW_TRACK_MAP = TOPOLOGY_DATA["raw"]["track_map"]

TRAIN_STATES: Dict[str, Any] = {}
_INFERENCE_TRAIN_IDS: List[str] = []  # Ordered IDs for RL obs consistency
ACTIVE_WEBSOCKETS: set = set()          # Topology broadcast sockets
COPILOT_WEBSOCKETS: set = set()         # AI Co-pilot broadcast sockets

# System Overrides
SYSTEM_LOCKDOWN = False
OR_SHIELD_ENABLED = True
AI_AUTO_COMMIT = False      # kept for backward-compat reads
AUTOPILOT_MODE = True       # kept for backward-compat reads
EXPLAIN_BEFORE_ACT_MODE = False  # when True: contested decisions HOLD pending override-or-confirm
SIM_SPEED_FACTOR = 0.4
TICK_INTERVAL_S = 1.0

# ---------------------------------------------------------------------------
# Operator-loop action tables (Change #1–#3)
# ---------------------------------------------------------------------------
# LATEST_MODEL_PROPOSAL  — read-only advisory copy of the last model prediction.
#   Written after model.predict(), never directly applied to the sim.
LATEST_MODEL_PROPOSAL: Dict[str, int] = {}

# PENDING_OPERATOR_ACTIONS — one-shot actions injected by operator commit.
#   Consumed (popped) on the very next env.step() call.
PENDING_OPERATOR_ACTIONS: Dict[str, int] = {}

# STICKY_ACTIONS — persistent actions (e.g. STOP for N ticks).
#   Maps train_id → (action, expires_tick).  Checked before PENDING_OPERATOR_ACTIONS.
STICKY_ACTIONS: Dict[str, Tuple[int, int]] = {}  # train_id → (action, expires_tick)

# How many ticks a committed STOP action persists before the train resumes MAIN.
OVERRIDE_TICKS = 15

# OR-Shield singleton — validates proposals before queuing + at commit time
_OR_SHIELD = SmartOptimizer()


# ---------------------------------------------------------------------------
# In-memory suggestion cache  (recommendation_id → AISuggestion dict)
# ---------------------------------------------------------------------------
COPILOT_SUGGESTIONS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# In-memory maintenance block store  (element_id → block dict)
# ---------------------------------------------------------------------------
ACTIVE_BLOCKS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Dynamic Constraints from Sandbox (constraint_id -> constraint dict)
# ---------------------------------------------------------------------------
DYNAMIC_CONSTRAINTS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Simulation Brain — lazy PPO model singleton
# ---------------------------------------------------------------------------
_SIM_MODEL = None
_SIM_ENV   = None
INFERENCE_ACTIVE = False
_INFERENCE_OBS = None
_INFERENCE_ACTIONS = None  # raw numpy array from model.predict (shape: [n_trains])
_INFERENCE_RAW_ACTIONS = None # original RL actions before dispatcher overrides
_INFERENCE_DECISION_META = {} # meta info for contested decisions

# ---------------------------------------------------------------------------
# Simulation tick counter — incremented once per simulate_trains_bg iteration.
# Used by the committed-override mechanism to enforce a time-bounded hold.
# ---------------------------------------------------------------------------
_SIM_TICK: int = 0
SUGGESTION_TTL_TICKS = 20
COPILOT_SUGGESTIONS_MAX_SIZE = 100

# Stores the result of the last successful OR-Tools schedule generation.
# Shape: { train_id: { node_id: {arrival, departure} } }
LAST_OR_SCHEDULE: Dict[str, Any] = {}

def _get_sim_brain():
    """
    Lazily load the trained PPO model + a fresh TrainDispatchEnv on first call.
    Subsequent calls return the cached singleton.
    Returns (model, env) or (None, None) if the model file is missing.
    """
    global _SIM_MODEL, _SIM_ENV
    if _SIM_MODEL is not None:
        return _SIM_MODEL, _SIM_ENV

    # ── Level 6 checkpoint (Latest best) ──────────────────
    model_path = os.path.join(
        os.path.dirname(__file__), "ai", "models", "Phase3", "L6_25Trains_Best_v4", "best_model.zip"
    )
    stats_path = os.path.join(
        os.path.dirname(__file__), "ai", "models", "Phase3", "vec_normalize_L6_25Trains.pkl"
    )

    if not os.path.exists(model_path):
        print(f"[SIM-BRAIN] ⚠️  Model not found at {model_path} — falling back to OR-Tools only.")
        return None, None

    try:
        os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from train_env import TrainDispatchEnv

        # Build env at 25-train difficulty (must match training config)
        def make_env():
            e = TrainDispatchEnv()
            e.set_difficulty(25)
            return e

        raw_env = DummyVecEnv([make_env])

        # Load normalization stats if available (critical for correct inference)
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, raw_env)
            env.training = False   # freeze stats — do NOT update during inference
            env.norm_reward = False
            print("[SIM-BRAIN] 📊 VecNormalize stats loaded for 15-train model.")
        else:
            env = raw_env
            print("[SIM-BRAIN] ⚠️  No VecNormalize stats found — running without normalization.")

        model = MaskablePPO.load(
            model_path,
            env=None,   # pass obs manually; avoids obs-space shape conflict
            device="cpu"
        )
        _SIM_MODEL = model
        _SIM_ENV   = env
        print("✅ [SIM-BRAIN] 25-Train MaskablePPO model (best checkpoint) loaded for sandbox analysis.")
        return model, env
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[SIM-BRAIN] ❌ Failed to load model: {exc}")
        return None, None

# ---------------------------------------------------------------------------
# In-memory Fleet Registry  (train_id → fleet config dict)
# Seeded at startup from the simulation TRAIN_STATES; editable via REST API.
# ---------------------------------------------------------------------------
FLEET_REGISTRY: Dict[str, Any] = {}

TRAIN_TYPES  = ["Vande Bharat", "Rajdhani", "Superfast", "Express",
                "Local", "Suburban", "Passenger", "Freight (WAG-9)"]
PRIORITY_MAP = {
    "Vande Bharat": 10, "Rajdhani": 10, "Superfast": 8,
    "Express": 6, "Local": 5, "Suburban": 5,
    "Passenger": 3, "Freight (WAG-9)": 2,
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class RejectRequest(BaseModel):
    recommendation_id: str
    reason: Optional[str] = "controller_dismissed"

class InfrastructureBlock(BaseModel):
    element_id: str                                    # e.g. "edge-0-1"
    type: Literal["PLATFORM", "TRACK_SEGMENT"]         # block type
    start_time: str                                    # ISO-8601
    end_time: str                                      # ISO-8601
    severity: Literal["TOTAL_BLOCK", "SPEED_RESTRICTION"]
    reason: Optional[str] = "Scheduled maintenance"

class NewTrainRequest(BaseModel):
    train_id: str
    train_type: str = "Express"           # must be one of TRAIN_TYPES
    max_speed: int  = 110                 # km/h
    priority: Optional[int] = None        # auto-derived from type if omitted
    start_time: int = 0                   # minutes from session start
    deadline: int   = 120                 # minutes from session start
    direction: int  = 1                   # 1 = forward

class SimSpeedRequest(BaseModel):
    factor: float

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _sync_blocks_to_rl_env() -> None:
    """
    Patches the RL model's inner track_map to temporarily remove edges that are
    under a TOTAL_BLOCK maintenance window. This prevents the model from issuing
    MAIN actions toward blocked segments during live inference.
    Called any time ACTIVE_BLOCKS changes.
    """
    global _SIM_ENV
    if _SIM_ENV is None or not INFERENCE_ACTIVE:
        return
    try:
        inner_env = _SIM_ENV.venv.envs[0] if hasattr(_SIM_ENV, 'venv') else _SIM_ENV.envs[0]
        # Start from the canonical topology
        import copy
        patched_map = copy.deepcopy(RAW_TRACK_MAP)

        for edge_id, block in ACTIVE_BLOCKS.items():
            if block.get("severity") != "TOTAL_BLOCK":
                continue
            parts = edge_id.split("-")
            if len(parts) < 3:
                continue
            try:
                src = int(parts[1])
                dst = int(parts[2])
            except ValueError:
                continue
            # Remove the blocked destination from the source node's 'next' list
            if src in patched_map and dst in patched_map[src].get("next", []):
                patched_map[src]["next"] = [
                    n for n in patched_map[src]["next"] if n != dst
                ]
            
            # Remove the blocked source from the destination node's 'prev' list
            if dst in patched_map and src in patched_map[dst].get("prev", []):
                patched_map[dst]["prev"] = [
                    n for n in patched_map[dst]["prev"] if n != src
                ]
        inner_env.track_map = patched_map
    except Exception as e:
        print(f"[REROUTE] ⚠️  RL env track_map sync failed: {e}")

def _reroute_live_train(t_id: str, blocked_edge_id: str) -> bool:
    """
    Attempt to rewrite the live path of a single train in TRAIN_STATES,
    replacing a blocked edge segment with a clear alternate.
    Returns True if a valid bypass was found and applied, False otherwise.
    """
    state = TRAIN_STATES.get(t_id)
    if not state:
        return False

    original_path = state.get("path", [])
    if blocked_edge_id not in original_path:
        return False

    # Find the index of the blocked edge in the path
    try:
        block_idx = original_path.index(blocked_edge_id)
    except ValueError:
        return False

    # Parse the source node of the blocked edge
    parts = blocked_edge_id.split("-")
    if len(parts) < 3:
        return False

    try:
        src_node = int(parts[1])
    except ValueError:
        return False

    # Junction bypass: try alternate siblings from RAW_TRACK_MAP
    options = RAW_TRACK_MAP.get(src_node, {}).get("next", [])
    for alt_dst in options:
        alt_edge_in = f"edge-{src_node}-{alt_dst}"
        if alt_edge_in == blocked_edge_id or alt_edge_in in ACTIVE_BLOCKS:
            continue
        # Check the exit from the alternate node is also clear
        exits = RAW_TRACK_MAP.get(alt_dst, {}).get("next", [])
        if not exits:
            continue
        alt_edge_out = f"edge-{alt_dst}-{exits[0]}"
        if alt_edge_out in ACTIVE_BLOCKS:
            continue

        # Build the new path: everything before the block + bypass + rest
        # The blocked edge + its usual exit (block_idx+1) are replaced by the two bypass edges
        before = original_path[:block_idx]
        after_skip = 1  # skip only the blocked edge; the exit from alt is different
        after = original_path[block_idx + after_skip + 1:] if block_idx + after_skip + 1 < len(original_path) else []
        new_path = before + [alt_edge_in, alt_edge_out] + after

        state["path"] = new_path
        print(f"[REROUTE] ✅ {t_id}: '{blocked_edge_id}' bypassed via '{alt_edge_in}'→'{alt_edge_out}'")
        return True

    return False


def _resolve_reroute_strategy(element_id: str) -> Dict[str, Any]:
    """
    Real OR-Tools reroute resolver — determines the best strategy when a
    maintenance block is applied and immediately mutates live TRAIN_STATES paths.

    Priority:
      1. SPATIAL_SHIFT   — an alternate parallel edge exists; path is rewritten
      2. TEMPORAL_SHIFT  — train is already past the block or no viable alternate
      3. MANUAL_INTERVENTION — no viable path; flag for human
    """
    blocked_trains = [
        t["train_id"]
        for t in TRAIN_STATES.values()
        if element_id in t.get("path", [])
    ]
    n_affected = len(blocked_trains)

    rerouted: List[str] = []
    halted_only: List[str] = []
    manual: List[str] = []

    for t_id in blocked_trains:
        state = TRAIN_STATES.get(t_id, {})
        curr_edge = state.get("edge_id", "")

        # If the train is already ON the blocked edge: halt it in place
        if curr_edge == element_id:
            state["status"] = "Blocked"
            # Still try to reroute the remaining path
            if _reroute_live_train(t_id, element_id):
                rerouted.append(t_id)
            else:
                halted_only.append(t_id)
        else:
            # Train hasn't reached the block yet — try spatial reroute
            if _reroute_live_train(t_id, element_id):
                rerouted.append(t_id)
            else:
                # No bypass available — halt the train before it reaches the block
                state["status"] = "Halted"
                halted_only.append(t_id)

    # Also patch the RL inner env track_map so the model stops routing toward blocked edge
    _sync_blocks_to_rl_env()

    if not blocked_trains:
        strategy = "TEMPORAL_SHIFT"
        detail = "No trains with this segment in their path. Block applied with zero immediate impact."
    elif rerouted:
        strategy = "SPATIAL_SHIFT"
        detail = (
            f"{len(rerouted)} train(s) dynamically rerouted: {rerouted}. "
            + (f"{len(halted_only)} train(s) halted (no bypass): {halted_only}." if halted_only else "")
        )
    elif halted_only:
        strategy = "TEMPORAL_SHIFT"
        detail = (
            f"{len(halted_only)} train(s) halted — no alternate path found. "
            "Trains will wait until the block is cleared."
        )
    else:
        strategy = "MANUAL_INTERVENTION"
        detail = f"{n_affected} train(s) require manual dispatch intervention."

    return {
        "affected_trains": n_affected,
        "affected_train_ids": blocked_trains,
        "rerouted_train_ids": rerouted,
        "halted_train_ids": halted_only,
        "strategy": strategy,
        "detail": detail,
        "element_id": element_id,
        "timestamp": _now_iso(),
    }
def _compute_impact_minutes(train_id: str, rl_action: int) -> int:
    """
    Returns estimated delay impact in minutes.
    Positive = time saved. Negative = delay added.
    Reads from LAST_OR_SCHEDULE if available, else uses physics estimate.
    """
    state = TRAIN_STATES.get(train_id)
    if not state:
        return 0

    scheduled_arrival = state.get("scheduled_arrival", 0)
    sim_time          = state.get("sim_time", 0)
    delay_so_far      = state.get("delay_mins", 0)

    if rl_action == 0:  # STOP — adds roughly 1-2 minutes of waiting
        return -2       # negative: costs the train 2 minutes
    elif rl_action == 2:  # DIVERT — loop adds ~3-5 min but prevents collision penalty
        path = state.get("path", [])
        trains_on_mainline = sum(
            1 for s in TRAIN_STATES.values()
            if s.get("edge_id") in path and s.get("train_id") != train_id
        )
        return max(3, trains_on_mainline * 2)  # positive: saves cascade delays

    return 0


def _make_suggestion() -> list:
    """Returns a list of AI recommendations (empty list if none)."""
    global INFERENCE_ACTIVE, _INFERENCE_ACTIONS, _INFERENCE_ACTION_PROBS, _INFERENCE_RAW_ACTIONS

    if not (INFERENCE_ACTIVE and _INFERENCE_RAW_ACTIONS is not None):
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

    for i, act in enumerate(_INFERENCE_RAW_ACTIONS):
        if i >= len(_INFERENCE_TRAIN_IDS):
            break

        tid = _INFERENCE_TRAIN_IDS[i]
        state = TRAIN_STATES.get(tid)

        # Guard 1: train not in TRAIN_STATES yet (startup race) or fully inactive
        if not state or state.get("status") in _INACTIVE_STATUSES:
            continue

        meta = _INFERENCE_DECISION_META.get(tid, {})
        is_contested = meta.get("contested", False)
        if not is_contested:
            continue

        edge_id = state.get("edge_id", "edge-0-1")

        # Guard 2: train is still on a staging edge (status may lag by one tick)
        if edge_id in _STAGING_EDGES:
            continue

        # Guard 3: skip trains that already have an active committed STOP sticky action.
        # Prevents flooding the panel with STOP suggestions for trains the operator
        # has already told to hold.
        sticky = STICKY_ACTIONS.get(tid)
        if act == 0 and sticky and sticky[1] > _SIM_TICK:
            continue

        # Safe probability extraction
        try:
            prob = float(_INFERENCE_ACTION_PROBS[i]) if _INFERENCE_ACTION_PROBS and i < len(_INFERENCE_ACTION_PROBS) else 0.85
        except Exception:
            prob = 0.85
        prob = max(0.0, min(1.0, prob))

        # Check if any other active train is on the same edge
        next_edge_occupied = any(
            s.get("edge_id") == edge_id and k != tid
            for k, s in TRAIN_STATES.items()
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
                "impact_analysis"   : _compute_impact_minutes(tid, act),
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
                "decided_at_tick"   : _SIM_TICK,
                "obs_snapshot"      : {
                    "edge_id"            : edge_id,
                    "position_percentage": state.get("position_percentage", 0),
                    "speed_kmh"          : state.get("speed_kmh", 0),
                    "status"             : state.get("status"),
                    "sim_time"           : state.get("sim_time", 0),
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
                "impact_analysis"   : _compute_impact_minutes(tid, act),
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
                "decided_at_tick"   : _SIM_TICK,
                "obs_snapshot"      : {
                    "edge_id"            : edge_id,
                    "position_percentage": state.get("position_percentage", 0),
                    "speed_kmh"          : state.get("speed_kmh", 0),
                    "status"             : state.get("status"),
                    "sim_time"           : state.get("sim_time", 0),
                },
            })

        # ── Action = MAIN (1): Only suggest if train is actively stopped ──────
        elif act == 1:
            # Only meaningful if the train is stopped AND actively in service
            # (not just waiting to spawn — those have speed=0 too)
            is_stopped = state.get("speed_kmh", 0) < 5
            is_active  = state.get("status") in ("Waiting at Signal", "Halted", "Blocked")
            if not is_stopped or not is_active:
                continue

            priority = 3
            urgency = "ADVISORY"
            action_str = f"Resume speed for {tid} to clear block"
            reasoning = "Path ahead is clear. Train should resume normal operating speed."
            suggestions.append({
                "recommendation_id" : str(uuid.uuid4()),
                "type"              : "AI_DECISION",
                "priority_level"    : priority,
                "urgency"           : urgency,
                "target_train_id"   : tid,
                "decided_action"    : action_str,
                "impact_analysis"   : _compute_impact_minutes(tid, act),
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
                "decided_at_tick"   : _SIM_TICK,
                "obs_snapshot"      : {
                    "edge_id"            : edge_id,
                    "position_percentage": state.get("position_percentage", 0),
                    "speed_kmh"          : state.get("speed_kmh", 0),
                    "status"             : state.get("status"),
                    "sim_time"           : state.get("sim_time", 0),
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
        except Exception:
            COPILOT_WEBSOCKETS.discard(ws)

async def _broadcast_topology(payload: Dict[str, Any]) -> None:
    """Send a message to all connected topology WebSocket clients."""
    text = json.dumps(payload)
    for ws in list(ACTIVE_WEBSOCKETS):
        try:
            await ws.send_text(text)
        except Exception:
            ACTIVE_WEBSOCKETS.discard(ws)

# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------
async def simulate_trains_bg():
    global TRAIN_STATES, INFERENCE_ACTIVE, _INFERENCE_OBS, _INFERENCE_ACTIONS, _SIM_TICK, _INFERENCE_DECISION_META
    from ai.config import generate_daily_schedule

    TRAIN_STATES = {}
    _spawned = False

    # Real mainline has nodes 0→1→2→...→83→999 (84 edges DOWN)
    DOWN_PATH = [f"edge-{k}-{k+1}" for k in range(83)] + ["edge-83-999"]
    # For UP trains, they traverse the exact same edges but in reverse order.
    # The UI only knows about forward edge IDs (edge-0-1, etc), so we must use those.
    UP_PATH   = ["edge-83-999"] + [f"edge-{k}-{k+1}" for k in reversed(range(83))]

    while True:
        if not INFERENCE_ACTIVE:
            await asyncio.sleep(1.0)
            continue
            
        _SIM_TICK += 1
        now_ts = time.time()
        # TTL Pruning for DYNAMIC_CONSTRAINTS
        expired = [c_id for c_id, c in list(DYNAMIC_CONSTRAINTS.items()) if c.get('expires_at', float('inf')) < now_ts]
        for c_id in expired:
            del DYNAMIC_CONSTRAINTS[c_id]
            AUDIT_LOGS.insert(0, {
                "t": _now_iso(),
                "source": "SIMULATION_ENGINE",
                "action": f"Constraint {c_id} automatically expired",
                "operator": "SYSTEM",
                "status": "Expired",
                "statusType": "info"
            })
        if SYSTEM_LOCKDOWN:
            for t_id, state in TRAIN_STATES.items():
                state['status'] = 'Halted'
        else:
            # ── Build the ordered list of active trains once per tick ──────────
            live_train_ids = list(TRAIN_STATES.keys())

            if INFERENCE_ACTIVE:
                try:
                    model, env = _get_sim_brain()
                    if model and env:
                        inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]

                        # ── Get observation and predict ────────────────────────
                        if _INFERENCE_OBS is None:
                            _INFERENCE_OBS = env.reset()

                        # Pass global sim speed to environment
                        if hasattr(inner_env, 'sim_speed_factor'):
                            pass
                        inner_env.sim_speed_factor = SIM_SPEED_FACTOR

                        action_masks = np.array(env.env_method("get_action_mask"))
                        action, _ = model.predict(
                            _INFERENCE_OBS, deterministic=True, action_masks=action_masks
                        )

                        import torch
                        obs_tensor = model.policy.obs_to_tensor(_INFERENCE_OBS)[0]
                        act_list = list(action[0]) if hasattr(action[0], '__iter__') else list(action)
                        global _INFERENCE_ACTION_PROBS
                        try:
                            with torch.no_grad():
                                dist = model.policy.get_distribution(obs_tensor)
                                action_tensor = torch.tensor(act_list).to(model.device)
                                log_probs = dist.log_prob(action_tensor)
                                probs = torch.exp(log_probs).cpu().numpy()
                            _INFERENCE_ACTION_PROBS = list(probs)
                        except Exception as e:
                            print(f"[ORBIT] ⚠️  Inference probability extraction error: {e}")
                            _INFERENCE_ACTION_PROBS = [0.85] * len(act_list)


                        global _INFERENCE_RAW_ACTIONS
                        raw_actions = list(action[0]) if hasattr(action[0], '__iter__') else list(action)
                        _INFERENCE_RAW_ACTIONS = raw_actions

                        # ── Advisory stage (read-only) ────────────────────────
                        # Store the model's full proposal without applying it.
                        # This is the source of truth for the transparency endpoint
                        # and for AUTOPILOT_MODE — it never directly touches TRAIN_STATES.
                        for _i, _t_id in enumerate(_INFERENCE_TRAIN_IDS):
                            if _i < len(raw_actions):
                                LATEST_MODEL_PROPOSAL[_t_id] = int(raw_actions[_i])

                        # ── Execution stage: autonomous-by-default, override-on-top ─────
                        #   1. STICKY_ACTIONS            — controller override, persists N ticks
                        #   2. PENDING_OPERATOR_ACTIONS  — controller override, one-shot
                        #   3. raw_actions[i]            — the model's own decision (DEFAULT — always active)
                        desired_actions = []
                        for i in range(len(raw_actions)):
                            t_id = _INFERENCE_TRAIN_IDS[i] if i < len(_INFERENCE_TRAIN_IDS) else ""
                            sticky = STICKY_ACTIONS.get(t_id)
                            if sticky and sticky[1] > _SIM_TICK:
                                desired_actions.append(sticky[0])
                            elif t_id in PENDING_OPERATOR_ACTIONS:
                                desired_actions.append(PENDING_OPERATOR_ACTIONS.pop(t_id))
                            else:
                                desired_actions.append(raw_actions[i])   # model's decision

                        # Step 2: OR-Shield validates the intent to prevent crashes
                        safe_actions, decision_meta = _OR_SHIELD.optimize_decision(
                            trains=inner_env.trains,
                            ai_actions=desired_actions,
                            track_map=inner_env.track_map,
                        )
                        
                        # EXPLAIN_BEFORE_ACT_MODE logic
                        if EXPLAIN_BEFORE_ACT_MODE:
                            for i, t in enumerate(inner_env.trains):
                                if decision_meta.get(t['id'], {}).get('contested', False):
                                    safe_actions[i] = 0  # Hold contested decisions
                                    
                        _INFERENCE_ACTIONS = safe_actions
                        _INFERENCE_DECISION_META = decision_meta

                        # Step 3: Execute safe actions in physics engine
                        step_actions = np.array([safe_actions])
                        _INFERENCE_OBS, _, terminated, _ = env.step(step_actions)[:4]

                        # ── Detect removed trains (finished/deadlocked) ────────
                        # The physics engine removes finished trains from its active list.
                        # We must catch this and mark them as Finished in the live map,
                        # otherwise they stay stuck as 'Moving' forever and inflate traffic counts.
                        current_rl_train_ids = {t['id'] for t in inner_env.trains}
                        for t_id, live in list(TRAIN_STATES.items()):
                            if t_id not in current_rl_train_ids and live.get('status') not in ('Finished', 'Scheduled', 'Expired'):
                                live['status'] = 'Finished'
                                live['finish_time'] = _SIM_TICK
                                live['speed_kmh'] = 0
                                # Move off-screen visually
                                dir_val = live.get('direction', 'DOWN')
                                if dir_val == "UP" or dir_val == 1:
                                    live['edge_id'] = "edge-0-1"
                                    live['position_percentage'] = 0.0
                                else:
                                    live['edge_id'] = "edge-83-999"
                                    live['position_percentage'] = 1.0

                        # ── Read RL env train positions back into TRAIN_STATES ─
                        # The RL env manages its own complete, valid train state.
                        # We map RL node positions → edge IDs for the live map.
                        # Real topology: nodes 0..83 → edge-{n}-{n+1}, node 83 → edge-83-999
                        for i, t_id in enumerate(_INFERENCE_TRAIN_IDS):
                            if t_id not in TRAIN_STATES or i >= len(inner_env.trains):
                                continue
                            rl_train = inner_env.trains[i]
                            live     = TRAIN_STATES[t_id]

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
                                live['finish_time'] = _SIM_TICK
                            elif node_id in (0, 998):
                                live['status'] = 'Scheduled'
                            elif rl_train.get('banker_wait', 0) > 0:
                                live['status'] = 'Banker Ops'
                            elif rl_train.get('dwell_rem', 0) > 0:
                                live['status'] = 'Boarding'
                            elif i < len(_INFERENCE_ACTIONS) and _INFERENCE_ACTIONS[i] == 0:
                                live['status'] = 'Waiting at Signal'
                            else:
                                live['status'] = 'Moving'

                        if bool(terminated[0]) if hasattr(terminated, '__getitem__') else bool(terminated):
                            # Auto-reset the RL env to keep inference running continuously.
                            # The internal RL episode ends when all trains arrive, but the
                            # live dashboard keeps going with newly scheduled trains.
                            print("[ORBIT] 🔄 RL episode complete — auto-resetting for continuous inference.")
                            _INFERENCE_OBS = env.reset()
                            _INFERENCE_RAW_ACTIONS = None

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[ORBIT] ⚠️  Inference sync error: {e}")

            # ── Drive map movement for ALL trains (inference + fallback) ───────
            # When inference is active, RL action determines speed (stop vs move).
            # When not active, all Moving trains advance normally.
            for idx, t_id in enumerate(live_train_ids):
                state = TRAIN_STATES[t_id]

                if state['status'] == 'Finished':
                    continue

                curr_edge = state.get('edge_id', '')

                # ── MAINTENANCE BLOCK CHECK (current edge) ────────────────────
                edge_block = ACTIVE_BLOCKS.get(curr_edge)
                if edge_block and edge_block.get('severity') == 'TOTAL_BLOCK':
                    state['status'] = 'Blocked'
                    continue

                # ── LOOKAHEAD BLOCK CHECK (next edge in path) ─────────────────
                # Prevent trains from advancing into a blocked segment
                path = state.get('path', [])
                try:
                    curr_path_idx = path.index(curr_edge)
                    if curr_path_idx + 1 < len(path):
                        next_edge = path[curr_path_idx + 1]
                        next_block = ACTIVE_BLOCKS.get(next_edge)
                        if next_block and next_block.get('severity') == 'TOTAL_BLOCK':
                            # Halt the train before it crosses into the blocked segment
                            state['status'] = 'Halted'
                            continue
                except (ValueError, IndexError):
                    pass

                # ── Per-train simulation clock ────────────────────────────────
                # Ticks up once per loop iteration for every active train.
                # Used for schedule-deadline comparisons and delay reporting.
                state['sim_time'] = state.get('sim_time', 0) + 1

                # Determine whether this train should move this tick
                if INFERENCE_ACTIVE and _INFERENCE_ACTIONS is not None:
                    # RL says: 0=STOP, 1=MAIN (move), 2=DIVERT (move to loop, treated as move)
                    rl_act = _INFERENCE_ACTIONS[idx] if idx < len(_INFERENCE_ACTIONS) else 1

                    # ── Auto-Intervention (Fix 7) ─────────────────────────────
                    if rl_act == 0 and state.get('override_expires', 0) <= _SIM_TICK and state.get('status') not in ('Finished', 'Scheduled'):
                        edge_id = state.get("edge_id")
                        next_edge_occupied = any(
                            s.get("edge_id") == edge_id and s.get("train_id") != t_id
                            for s in TRAIN_STATES.values()
                            if s.get("status") not in ("Finished", "Scheduled")
                        )
                        if next_edge_occupied:
                            state['override_action'] = 0
                            state['override_expires'] = _SIM_TICK + 8
                            asyncio.create_task(_broadcast_copilot({
                                "type": "AUTO_INTERVENTION",
                                "target_train_id": t_id,
                                "message": f"Critical intervention: Auto-applying STOP for {t_id} to prevent collision."
                            }))
                            AUDIT_LOGS.insert(0, {
                                "t": _now_iso(),
                                "source": "OR-SHIELD",
                                "action": f"Auto-applied STOP for {t_id}",
                                "operator": "SYSTEM",
                                "status": "Committed",
                                "statusType": "warning"
                            })

                    # ── Committed override takes priority over live RL ────────
                    # A controller commit writes override_action + override_expires
                    # to TRAIN_STATES.  While the override is active we honour the
                    # committed decision; once it expires the RL agent resumes.
                    override_exp = state.get('override_expires', 0)
                    if override_exp > _SIM_TICK:
                        rl_act = state['override_action']
                        ticks_left = override_exp - _SIM_TICK
                        print(f"[ORBIT] 🔒 Override active for {t_id}: "
                              f"action={rl_act} ({ticks_left} ticks remaining)")

                    should_move = (rl_act != 0)
                else:
                    should_move = (state['status'] in ('Moving', 'Blocked'))

                if should_move:
                    state['status'] = 'Moving'
                    # Only apply manual fallback movement if inference is not driving the positions
                    if not INFERENCE_ACTIVE:
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
                                    state['finish_time'] = _SIM_TICK
                            except ValueError:
                                pass
                else:
                    state['status'] = 'Halted'


        edges_occupied: Dict[str, list] = {}
        for t_id, state in TRAIN_STATES.items():
            if state.get('status') != 'Finished':
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
                    if t['status'] not in ('Blocked',):
                        t['status'] = 'Moving'

        payload = {
            "type": "topology_update",
            "trains": list(TRAIN_STATES.values()),
            "conflicts": list(conflicts),
            "maintenance_blocks": list(ACTIVE_BLOCKS.values()),
        }
        await _broadcast_topology(payload)
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
            if v.get("status") == "pending" and _SIM_TICK - v.get("suggested_at_tick", _SIM_TICK) > SUGGESTION_TTL_TICKS:
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

        if COPILOT_WEBSOCKETS and INFERENCE_ACTIVE:
            candidates = _make_suggestion()          # RL proposal
            if not candidates:
                await asyncio.sleep(2)
                continue
                
            for candidate in candidates:
                # ── OR-Shield Gate ──────────────────────────────────────────────
                if OR_SHIELD_ENABLED:
                    is_safe, reason = _OR_SHIELD.or_shield_check(
                        suggestion=candidate,
                        train_states=TRAIN_STATES,
                        active_blocks=ACTIVE_BLOCKS,
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
                    ticks_remaining = SUGGESTION_TTL_TICKS - (_SIM_TICK - candidate.get("suggested_at_tick", _SIM_TICK))
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



@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulate_trains_bg())
    asyncio.create_task(copilot_suggestion_bg())

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# System Override Endpoints
# ---------------------------------------------------------------------------
class OverrideRequest(BaseModel):
    enabled: bool

class WhatIfScenarioRequest(BaseModel):
    label: Optional[str] = "Scenario"
    delay_train_id: Optional[str] = ""
    latency_minutes: Optional[int] = 15
    forced_actions: Optional[Dict[str, int]] = {}   # train_id -> 0=HOLD, 1=MAIN, 2=DIVERT

@app.post("/api/v1/system/start-inference", tags=["System Override"])
async def start_inference():
    """
    Start the RL inference loop.
    """
    global INFERENCE_ACTIVE, _INFERENCE_OBS, _INFERENCE_ACTIONS, _INFERENCE_RAW_ACTIONS, TRAIN_STATES, LAST_OR_SCHEDULE, _INFERENCE_TRAIN_IDS
    INFERENCE_ACTIVE = False
    _INFERENCE_OBS = None
    _INFERENCE_ACTIONS = None
    _INFERENCE_RAW_ACTIONS = None
    if not LAST_OR_SCHEDULE:
        return {
            "status" : "error",
            "message": "No schedule found. Please generate a conflict-free schedule on the Fleet Status page first."
        }

    model, env = _get_sim_brain()
    if not (model and env):
        return {"status": "error", "message": "RL model could not be loaded."}

    # ── Re-seed TRAIN_STATES from OR schedule ─────────────────────────────
    new_states: Dict[str, Any] = {}
    ordered_ids = []
    
    # Reconstruct canonical path edges for DOWN/UP (nodes 0..83)
    DOWN_PATH = ["edge-0-1"] + [f"edge-{k}-{k+1}" for k in range(1, 83)] + ["edge-83-999"]
    UP_PATH   = ["edge-83-999"] + [f"edge-{k}-{k+1}" for k in reversed(range(83))]

    for t_id, cfg in FLEET_REGISTRY.items():
        path = cfg.get("path", [])
        if not path:
            direction_str = "DOWN" if cfg.get("direction", 1) in (1, "DOWN") else "UP"
            path = DOWN_PATH if direction_str == "DOWN" else UP_PATH
            cfg["path"] = path

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
            # ── Schedule timing ─────────────────────────────────────────────
            "scheduled_departure"  : cfg.get("start_time", 0),
            "scheduled_arrival"    : cfg.get("deadline", 120),
            "delay_mins"           : 0,
            "sim_time"             : 0,
            "override_action"      : 1,
            "override_expires"     : 0,
        }

    if new_states:
        TRAIN_STATES = new_states
        _INFERENCE_TRAIN_IDS = ordered_ids

    # ── Reset RL env & activate inference ─────────────────────────────────
    # Use the custom schedule generated by OR-Tools so the RL env matches the UI.
    inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
    
    formatted_schedule = {}
    for t_id, t_sched in LAST_OR_SCHEDULE.items():
        cfg = FLEET_REGISTRY.get(t_id, {})
        formatted_schedule[t_id] = {
            'start_time': cfg.get("start_time", 0),
            'deadline': cfg.get("deadline", 120),
            'direction': "DOWN" if cfg.get("direction", 1) in (1, "DOWN") else "UP",
            'stops': list(t_sched.keys())
        }
        
    inner_env.set_custom_schedule(
        fleet=list(FLEET_REGISTRY.values()),
        schedule=formatted_schedule
    )
    _INFERENCE_OBS     = env.reset()
    _INFERENCE_ACTIONS = None
    INFERENCE_ACTIVE   = True

    print(f"[ORBIT] 🚀 Inference started. {len(TRAIN_STATES)} trains seeded from OR schedule.")
    return {
        "status" : "started",
        "active" : True,
        "trains" : len(TRAIN_STATES),
    }

@app.post("/api/v1/system/stop-inference", tags=["System Override"])
async def stop_inference():
    global INFERENCE_ACTIVE
    INFERENCE_ACTIVE = False
    return {"status": "stopped", "active": False}

@app.get("/api/v1/system/inference-status", tags=["System Override"])
async def get_inference_status():
    global INFERENCE_ACTIVE, OR_SHIELD_ENABLED, AI_AUTO_COMMIT, AUTOPILOT_MODE, SYSTEM_LOCKDOWN
    return {
        "active": INFERENCE_ACTIVE,
        "safety_shield": OR_SHIELD_ENABLED,
        "auto_commit": AUTOPILOT_MODE,   # backward-compat key — now mirrors AUTOPILOT_MODE
        "autopilot_mode": AUTOPILOT_MODE,
        "lockdown": SYSTEM_LOCKDOWN
    }

@app.post("/api/v1/system/sim-speed", tags=["System Controls"])
async def set_sim_speed(req: SimSpeedRequest):
    """
    Adjust simulation speed by updating the global sleep interval.
    The UI sends the literal interval in seconds (e.g. 0.2 for 5x speed).
    """
    global TICK_INTERVAL_S
    TICK_INTERVAL_S = max(0.05, min(req.factor, 5.0))
    return {"status": "success", "sim_speed": TICK_INTERVAL_S}

@app.post("/api/v1/system/lockdown", tags=["System Override"])
async def toggle_lockdown(req: OverrideRequest):
    global SYSTEM_LOCKDOWN
    SYSTEM_LOCKDOWN = req.enabled
    
    if not SYSTEM_LOCKDOWN:
        # Resume all trains that were halted by lockdown, unless they are blocked by an edge block
        for t_id, state in TRAIN_STATES.items():
            if state.get("status") == "Halted":
                state["status"] = "Moving"
                
    status_text = "ACTIVATED" if SYSTEM_LOCKDOWN else "DEACTIVATED"
    entry = {
        "t"         : _now_iso(),
        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
        "source"    : "SYSTEM_CONTROL",
        "action"    : f"Emergency Stop {status_text}",
        "operator"  : "Dispatcher",
        "status"    : "Lockdown" if SYSTEM_LOCKDOWN else "Nominal",
        "statusType": "error" if SYSTEM_LOCKDOWN else "success",
        "id"        : str(uuid.uuid4())
    }
    AUDIT_LOGS.insert(0, entry)
    _persist_audit_log(entry)

    return {"status": "success", "lockdown": SYSTEM_LOCKDOWN}

@app.post("/api/v1/system/safety-shield", tags=["System Override"])
async def toggle_safety_shield(req: OverrideRequest):
    global OR_SHIELD_ENABLED
    OR_SHIELD_ENABLED = req.enabled
    
    status_text = "ACTIVATED" if OR_SHIELD_ENABLED else "DEACTIVATED"
    entry = {
        "t"         : _now_iso(),
        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
        "source"    : "SYSTEM_CONTROL",
        "action"    : f"OR-Shield Safety Protocol {status_text}",
        "operator"  : "Dispatcher",
        "status"    : "Active" if OR_SHIELD_ENABLED else "Disabled",
        "statusType": "success" if OR_SHIELD_ENABLED else "warning",
        "id"        : str(uuid.uuid4())
    }
    AUDIT_LOGS.insert(0, entry)
    _persist_audit_log(entry)

    return {"status": "success", "safety_shield": OR_SHIELD_ENABLED}

@app.post("/api/v1/system/auto-commit", tags=["System Override"])
async def toggle_auto_commit_legacy(req: OverrideRequest):
    """Deprecated alias — use POST /api/v1/system/autopilot instead."""
    return await toggle_autopilot(req)

@app.post("/api/v1/system/autopilot", tags=["System Override"])
async def toggle_autopilot(req: OverrideRequest):
    """Toggle Autopilot Mode.

    When enabled, the execution stage applies the model's advisory proposal for
    every train that lacks an explicit operator-committed action.  This is a
    clearly-labelled, separately-toggled path — not a hidden side-effect of the
    suggestion generator.
    """
    global AI_AUTO_COMMIT, AUTOPILOT_MODE
    AUTOPILOT_MODE = req.enabled
    AI_AUTO_COMMIT = req.enabled   # keep legacy alias in sync

    status_text = "ACTIVATED" if AUTOPILOT_MODE else "DEACTIVATED"
    entry = {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SYSTEM_CONTROL",
        "action": f"Autopilot Mode {status_text}",
        "operator": "Dispatcher",
        "status": "Active" if AUTOPILOT_MODE else "Disabled",
        "statusType": "warning" if AUTOPILOT_MODE else "success",
        "id": str(uuid.uuid4())
    }
    AUDIT_LOGS.insert(0, entry)
    _persist_audit_log(entry)

    return {"status": "success", "autopilot_mode": AUTOPILOT_MODE, "auto_commit": AUTOPILOT_MODE}

AUDIT_LOGS = []

def _persist_audit_log(entry: dict):
    """Fire-and-forget DB write for an audit log entry.  Silently swallows errors
    so a DB hiccup never disrupts the hot simulation path."""
    try:
        db = database.SessionLocal()
        crud.create_audit_log(db, entry)
        db.close()
    except Exception as _e:
        print(f"[ORBIT] Warning: audit log DB write failed: {_e}")

@app.get("/api/v1/system/audit-logs", tags=["System Override"])
async def get_audit_logs(limit: int = 50, skip: int = 0):
    """Return audit log entries.  Merges in-memory (current session) with
    persisted DB records so history survives restarts."""
    try:
        db = database.SessionLocal()
        db_logs = crud.get_recent_audit_logs(db, limit=limit + len(AUDIT_LOGS), skip=0)
        db.close()
        # Convert ORM rows to dicts matching the in-memory format
        db_entries = [
            {
                "id": row.log_id,
                "t": row.timestamp,
                "timestamp": row.timestamp_ms,
                "source": row.source,
                "action": row.action,
                "operator": row.operator,
                "status": row.status,
                "statusType": row.status_type,
            }
            for row in db_logs
        ]
    except Exception:
        db_entries = []

    # Merge: in-memory entries take precedence (they're newer / more authoritative
    # for the current session); de-duplicate by id.
    seen_ids = set()
    merged = []
    for entry in AUDIT_LOGS + db_entries:
        eid = entry.get("id", "")
        if eid and eid in seen_ids:
            continue
        seen_ids.add(eid)
        merged.append(entry)

    merged.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {
        "logs": merged[skip : skip + limit],
        "total": len(merged)
    }


# ---------------------------------------------------------------------------
# DB Dependency
# ---------------------------------------------------------------------------
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Existing Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ORBIT: Operational Rail Backbone API is Running"}

@app.get("/train/logs", response_model=List[schema.TrainPosition], tags=["Trains"])
def get_train_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Return recent train position logs."""
    return crud.getRecentTrainLog(db, limit=limit)

@app.get("/simulation/live")
def get_live_simulation(db: Session = Depends(get_db)):
    live_data = []
    for t in TRAIN_CONFIG:
        train_id = t['id']
        latest = db.query(TrainPosition).filter(
            TrainPosition.train_id == train_id
        ).order_by(TrainPosition.id.desc()).first()
        if latest:
            live_data.append({
                "id": train_id,
                "section": int(latest.section) if hasattr(latest, 'section') and latest.section and str(latest.section).isdigit() else 0,
                "speed": latest.speed_kmh,
                "status": latest.Status,
                "priority": t['priority']
            })
    return live_data

# Pages 3 & 4: Optimization Logic
@app.post("/api/v1/simulate")
async def run_simulation(data: Dict[Any, Any]):
    return {"status": "optimized", "changes": []}

# Pages 0 & 5: Metadata
@app.post("/api/v1/telemetry")
async def post_telemetry(req: dict):
    print(f"FRONTLOG: {req}")
    return {"status": "ok"}

@app.get("/api/v1/meta")
async def get_meta():
    return {"version": "2.0", "author": "Swastik (MITS)"}

@app.get("/api/v1/telemetry", tags=["Telemetry"])
async def get_telemetry():
    """Returns real-time telemetry calculated from the current simulation state."""
    active_trains = 0
    incoming_trains = 0
    outgoing_trains = 0
    terminal_trains = 0
    halted_trains = 0

    for t_id, state in TRAIN_STATES.items():
        if state.get("status") == "Finished":
            terminal_trains += 1
            continue
            
        active_trains += 1
        
        # Determine direction based on state
        dir_val = state.get("direction")
        if dir_val is None:
            dir_val = FLEET_REGISTRY.get(t_id, {}).get("direction", "DOWN")
            
        if dir_val == "UP" or dir_val == 1:
            incoming_trains += 1
        else:
            outgoing_trains += 1
        
        if state.get("status") in ("Blocked", "Halted"):
            halted_trains += 1

    DELAY_THRESHOLD = 10.0
    on_time_trains = 0
    evaluated_trains = 0

    for t_id, state in TRAIN_STATES.items():
        reg = FLEET_REGISTRY.get(t_id, {})
        deadline = reg.get("deadline", 120)

        delay = 0.0
        if state.get("status") == "Finished":
            t_actual = state.get("finish_time", _SIM_TICK)
            delay = max(0.0, t_actual - deadline)
        else:
            path = state.get("path", [])
            curr_edge = state.get("edge_id")
            if path and curr_edge in path:
                path_len = len(path)
                curr_idx = path.index(curr_edge)
                edge_pct = state.get("position_percentage", 0.0)
                completion = (curr_idx + edge_pct) / path_len
                
                if completion > 0.01:
                    eta = _SIM_TICK / completion
                    delay = max(0.0, eta - deadline)
                else:
                    delay = 0.0

        if delay <= DELAY_THRESHOLD:
            on_time_trains += 1
            
        evaluated_trains += 1

    if evaluated_trains > 0:
        punctuality = (on_time_trains / evaluated_trains) * 100.0
    else:
        punctuality = 100.0

    halted_pct = (halted_trains / active_trains * 100) if active_trains > 0 else 0
    blocks_active = len(ACTIVE_BLOCKS) > 0
    
    if halted_pct > 20:
        network_fluidity = "Degraded"
    elif blocks_active or (10 <= halted_pct <= 20):
        network_fluidity = "Warning"
    else:
        network_fluidity = "Nominal"

    ai_load = 40 + (active_trains * 5) + (len(ACTIVE_BLOCKS) * 15)
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
        "schedule_ready": len(LAST_OR_SCHEDULE) > 0,
        "schedule_train_count": len(LAST_OR_SCHEDULE),
        "lockdown": SYSTEM_LOCKDOWN,
        "active": INFERENCE_ACTIVE,
        "timestamp": _now_iso()
    }

# ---------------------------------------------------------------------------
# WebSockets
# ---------------------------------------------------------------------------
@app.websocket("/ws/traffic")
async def traffic_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        print(f"Traffic WS disconnected: {e}")


@app.websocket("/ws/topology")
async def topology_websocket(websocket: WebSocket):
    await websocket.accept()
    ACTIVE_WEBSOCKETS.add(websocket)
    await websocket.send_text(json.dumps({
        "type": "topology_init",
        "topology": NETWORK_TOPOLOGY
    }))
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        ACTIVE_WEBSOCKETS.discard(websocket)
        print(f"Topology WS disconnected: {e}")


@app.websocket("/ws/copilot")
async def copilot_websocket(websocket: WebSocket):
    """
    ORBIT AI Co-pilot WebSocket.
    """
    await websocket.accept()
    COPILOT_WEBSOCKETS.add(websocket)
    
    pending = [s for s in COPILOT_SUGGESTIONS.values() if s.get("status") == "pending"]
    pending = [s for s in pending if _SIM_TICK - s.get("suggested_at_tick", _SIM_TICK) <= SUGGESTION_TTL_TICKS]
    
    for suggestion in pending[-3:]:
        try:
            await websocket.send_text(json.dumps(suggestion))
        except Exception:
            break

    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        COPILOT_WEBSOCKETS.discard(websocket)
        print(f"[ORBIT] Co-pilot WS disconnected: {e}")

# ---------------------------------------------------------------------------
# ORBIT Dispatch Endpoints
# ---------------------------------------------------------------------------
def _write_feedback(
    suggestion: Dict[str, Any],
    outcome: str,
    reason: str = "",
    original_action: Optional[int] = None,
    original_edge: Optional[str] = None,
):
    """Write RLHF telemetry to human_feedback.jsonl.

    When the operator modifies a suggestion before committing, pass the model's
    original proposal via ``original_action``/``original_edge`` so the diff is
    captured explicitly.  This is the most valuable training signal:
    "model proposed X, operator corrected to Y."
    """
    try:
        operator_action = suggestion.get("rl_action")
        operator_edge   = (suggestion.get("affected_edges") or [None])[0]
        was_modified = (
            original_action is not None and original_action != operator_action
        ) or (
            original_edge is not None and original_edge != operator_edge
        )
        feedback = {
            "timestamp"         : _now_iso(),
            "recommendation_id" : suggestion.get("recommendation_id"),
            "target_train_id"   : suggestion.get("target_train_id"),
            "proposed_action"   : suggestion.get("proposed_action"),
            "rl_action"         : suggestion.get("rl_action"),
            "obs_snapshot"      : suggestion.get("obs_snapshot", {}),
            "outcome"           : outcome,
            "reason"            : reason,
            # Modification diff — present only when operator changed the proposal
            "was_modified"      : was_modified,
            "original_rl_action": original_action,   # model's proposal before edit
            "operator_action"   : operator_action,    # what the operator chose
            "original_edge"     : original_edge,
            "operator_edge"     : operator_edge,
        }
        with open("human_feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback) + "\n")
    except Exception as e:
        print(f"[ORBIT] Warning: failed to write feedback: {e}")

class OverrideRequest(BaseModel):
    recommendation_id: str
    new_action: Optional[int] = None
    new_edge: Optional[str] = None

@app.post("/api/v1/dispatch/override", tags=["ORBIT Co-pilot"])
async def override_decision(req: OverrideRequest):
    suggestion = COPILOT_SUGGESTIONS.get(req.recommendation_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Decision not found or already expired.")

    if suggestion["status"] != "executed":
        raise HTTPException(
            status_code=409,
            detail=f"Decision is already '{suggestion['status']}'."
        )

    # Capture original values BEFORE applying operator modifications.
    # These are passed to _write_feedback so the diff is recorded explicitly.
    original_action: Optional[int] = suggestion.get("rl_action")
    original_edge: Optional[str]   = (suggestion.get("affected_edges") or [None])[0]

    if req.new_action is not None:
        suggestion["rl_action"] = req.new_action
    if req.new_edge is not None:
        suggestion["affected_edges"] = [req.new_edge]

    train_id = suggestion["target_train_id"]
    current_state = TRAIN_STATES.get(train_id)
    if not current_state:
        raise HTTPException(status_code=400, detail="Train no longer active.")

    suggested_tick = suggestion.get("suggested_at_tick", _SIM_TICK)
    if _SIM_TICK - suggested_tick > SUGGESTION_TTL_TICKS:
        suggestion["status"] = "expired"
        _write_feedback(suggestion, "expired", "Staleness TTL exceeded")
        raise HTTPException(status_code=400, detail="Suggestion has expired (TTL).")

    suggested_edge = suggestion.get("suggested_at_edge")
    if suggested_edge and current_state.get("edge_id") != suggested_edge:
        suggestion["status"] = "expired"
        _write_feedback(suggestion, "expired", "Positional staleness - train has moved.")
        raise HTTPException(status_code=400, detail="Train has already moved past the suggested decision point.")

    is_safe, reason = _OR_SHIELD.or_shield_check(
        suggestion=suggestion,
        train_states=TRAIN_STATES,
        active_blocks=ACTIVE_BLOCKS,
        dynamic_constraints=DYNAMIC_CONSTRAINTS,
    )

    if is_safe:
        for edge_id in suggestion.get("affected_edges", []):
            blk = ACTIVE_BLOCKS.get(edge_id)
            if blk and blk.get("severity") == "TOTAL_BLOCK":
                is_safe = False
                reason = (
                    f"MaintenanceBlock: edge '{edge_id}' is under an active "
                    "TOTAL_BLOCK maintenance window — dispatch blocked"
                )
                break

    if not is_safe:
        raise HTTPException(
            status_code=409,
            detail=f"SafetyConflict: {reason}. Dispatch blocked by OR-Shield."
        )

    # 4. Mark overridden
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
    AUDIT_LOGS.append(audit_entry)
    _persist_audit_log(audit_entry)

    # 5. Apply decision via clean action injection
    #
    #    The commit endpoint NO LONGER directly manipulates TRAIN_STATES physics
    #    (no more manual path splicing, no more edge advances, no more override_action keys).
    #
    #    Instead we inject intent into PENDING_OPERATOR_ACTIONS / STICKY_ACTIONS.
    #    The execution stage in simulate_trains_bg reads these on the very next
    #    env.step() call — env.step() is the single source of truth for all physics
    #    (token logic, capacity checks, banker logic, dwell logic).
    #
    #    STOP  → sticky: persists for OVERRIDE_TICKS ticks via STICKY_ACTIONS
    #    DIVERT → one-shot: env.step() handles loop selection via PENDING_OPERATOR_ACTIONS
    #    MAIN   → one-shot: proceed signal via PENDING_OPERATOR_ACTIONS
    target_id = suggestion["target_train_id"]
    rl_action = suggestion.get("rl_action", 1)  # 0=STOP, 1=MAIN, 2=DIVERT
    new_edge_id: Optional[str] = None

    if rl_action == 0:
        # STOP — sticky hold for OVERRIDE_TICKS ticks
        STICKY_ACTIONS[target_id] = (0, _SIM_TICK + OVERRIDE_TICKS)
        # Also update the live display status immediately so the map shows "Halted"
        if target_id in TRAIN_STATES:
            TRAIN_STATES[target_id]["status"] = "Halted"
        print(f"[COMMIT] 🛑 STOP sticky for {target_id} "
              f"({OVERRIDE_TICKS} ticks, expires tick {_SIM_TICK + OVERRIDE_TICKS})")

    elif rl_action == 2:
        # DIVERT — one-shot; env.step() picks the available loop node
        PENDING_OPERATOR_ACTIONS[target_id] = 2
        affected = suggestion.get("affected_edges", [])
        new_edge_id = affected[0] if affected else None
        print(f"[COMMIT] 🔀 DIVERT one-shot queued for {target_id}")

    else:
        # MAIN — one-shot proceed, explicitly clears any sticky holds
        PENDING_OPERATOR_ACTIONS[target_id] = 1
        STICKY_ACTIONS.pop(target_id, None)
        print(f"[COMMIT] ▶️  MAIN one-shot queued for {target_id}")

    # 5b. Write RLHF feedback with modification diff
    _write_feedback(
        suggestion,
        "overridden",
        "Operator overridden",
        original_action=original_action,
        original_edge=original_edge,
    )

    # 6. Broadcast SCHEDULE_UPDATED
    schedule_update = {
        "type"              : "SCHEDULE_UPDATED",
        "recommendation_id" : req.recommendation_id,
        "target_train_id"   : target_id,
        "decided_action"    : suggestion.get("decided_action", ""),
        "affected_edges"    : suggestion["affected_edges"],
        "new_edge_id"       : new_edge_id,
        "rl_action"         : rl_action,
        "overridden_at"     : overridden_at,
        "timestamp"         : overridden_at,
    }
    await _broadcast_topology(schedule_update)
    await _broadcast_copilot(schedule_update)

    print(
        f"[ORBIT] ✅ Overridden: {req.recommendation_id[:8]}… → "
        f"train {target_id} action={rl_action} "
        + (f"→ edge {new_edge_id}" if new_edge_id else "(queued for env.step)")
    )

    # 7. Return
    return {
        "status"            : "overridden",
        "recommendation_id" : req.recommendation_id,
        "target_train_id"   : target_id,
        "decided_action"    : suggestion.get("decided_action", ""),
        "new_edge_id"       : new_edge_id,
        "timestamp"         : overridden_at,
    }

class AcknowledgeRequest(BaseModel):
    recommendation_id: str
    reason: Optional[str] = "controller_dismissed"

@app.post("/api/v1/dispatch/acknowledge", tags=["ORBIT Co-pilot"])
async def acknowledge_decision(req: AcknowledgeRequest):
    """
    Human-dismissed: mark a recommendation as acknowledged/dismissed.
    """
    suggestion = COPILOT_SUGGESTIONS.get(req.recommendation_id)
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
    AUDIT_LOGS.append(audit_entry)
    _persist_audit_log(audit_entry)

    print(f"[ORBIT] 👁️  Acknowledged: {req.recommendation_id[:8]}…")

    return {
        "status"            : "acknowledged",
        "recommendation_id" : req.recommendation_id,
        "timestamp"         : suggestion["acknowledged_at"],
    }


@app.get("/api/v1/dispatch/suggestions", tags=["ORBIT Co-pilot"])
async def get_suggestions(status: Optional[str] = None):
    """List cached suggestions, optionally filtered by status."""
    results = list(COPILOT_SUGGESTIONS.values())
    if status:
        results = [s for s in results if s.get("status") == status]
    return {"suggestions": results, "count": len(results)}


# ---------------------------------------------------------------------------
# ORBIT Maintenance Management System (MMS) Endpoints
# ---------------------------------------------------------------------------
# [raw-proposal endpoint inserted above MMS section]

@app.get("/api/v1/copilot/raw-proposal", tags=["ORBIT Co-pilot"])
async def get_raw_proposal():
    """Return the model's full per-train action proposal — the unfiltered advisory output.

    The Co-Pilot panel surfaces at most 3 curated cards per cycle.
    This endpoint exposes the complete picture: what the model proposes for every
    active train, with confidence scores and current override state.

    Intended for: audit reviews, operator 'expand all' views, incident analysis.
    """
    action_labels = {0: "STOP", 1: "MAIN", 2: "DIVERT"}
    proposals = []
    for i, t_id in enumerate(_INFERENCE_TRAIN_IDS):
        model_action = LATEST_MODEL_PROPOSAL.get(t_id)
        sticky       = STICKY_ACTIONS.get(t_id)
        is_sticky    = bool(sticky and sticky[1] > _SIM_TICK)
        proposals.append({
            "train_id"                   : t_id,
            "model_action"               : model_action,
            "action_label"               : action_labels.get(model_action, "UNKNOWN"),
            "confidence"                 : (
                float(_INFERENCE_ACTION_PROBS[i])
                if _INFERENCE_ACTION_PROBS and i < len(_INFERENCE_ACTION_PROBS)
                else None
            ),
            "has_pending_operator_action": t_id in PENDING_OPERATOR_ACTIONS,
            "has_sticky_action"          : is_sticky,
            "sticky_action"              : action_labels.get(sticky[0], "UNKNOWN") if is_sticky else None,
            "sticky_expires_tick"        : sticky[1] if is_sticky else None,
            "ticks_until_sticky_expires" : (sticky[1] - _SIM_TICK) if is_sticky else None,
            "current_train_status"       : TRAIN_STATES.get(t_id, {}).get("status"),
            "current_edge"               : TRAIN_STATES.get(t_id, {}).get("edge_id"),
        })
    return {
        "tick"            : _SIM_TICK,
        "autopilot_mode"  : AUTOPILOT_MODE,
        "inference_active": INFERENCE_ACTIVE,
        "total_trains"    : len(_INFERENCE_TRAIN_IDS),
        "proposals"       : proposals,
    }

@app.get("/api/v1/maintenance/blocks", tags=["MMS"])
async def list_blocks():
    """Return all currently active infrastructure blocks."""
    return {"blocks": list(ACTIVE_BLOCKS.values()), "count": len(ACTIVE_BLOCKS)}


@app.post("/api/v1/maintenance/blocks", tags=["MMS"])
async def apply_block(block: InfrastructureBlock):
    """
    Apply an infrastructure maintenance block.

    Steps:
      1. Validate block doesn't duplicate an existing one.
      2. Store in ACTIVE_BLOCKS.
      3. Run OR-Tools reroute resolver.
      4. Broadcast MAINTENANCE_BLOCK_APPLIED to all topology WS clients.
      5. Return block + impact report.
    """
    # 1. Idempotent — update if exists
    block_dict = block.model_dump()
    block_dict["applied_at"] = _now_iso()
    ACTIVE_BLOCKS[block.element_id] = block_dict
    
    AUDIT_LOGS.append({
        "t": block_dict["applied_at"],
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": f"MMS_{block.element_id}",
        "action": f"Maintenance Applied: {block.severity} ({block.reason or 'Scheduled'})",
        "operator": "Dispatcher",
        "status": "Active",
        "statusType": "error",
        "id": str(uuid.uuid4())
    })

    # 2. Resolve reroute strategy
    impact = _resolve_reroute_strategy(block.element_id)

    # 3. Broadcast
    await _broadcast_topology({
        "type": "MAINTENANCE_BLOCK_APPLIED",
        "block": block_dict,
        "impact": impact,
    })
    await _broadcast_copilot({
        "type": "MAINTENANCE_BLOCK_APPLIED",
        "block": block_dict,
        "impact": impact,
    })

    n = impact["affected_trains"]
    strat = impact["strategy"].replace("_", " ").title()
    print(f"[MMS] 🔧 Block applied on '{block.element_id}' "
          f"({block.severity}) — {n} train(s) affected — strategy: {strat}")

    return {
        "status": "block_applied",
        "block": block_dict,
        "impact": impact,
        "timestamp": block_dict["applied_at"],
    }


@app.delete("/api/v1/maintenance/blocks/{element_id}", tags=["MMS"])
async def remove_block(element_id: str):
    """
    Remove an active maintenance block.
    Broadcasts MAINTENANCE_CLEARED to all WebSocket clients.
    """
    block = ACTIVE_BLOCKS.pop(element_id, None)
    if not block:
        raise HTTPException(
            status_code=404,
            detail=f"No active block found for element '{element_id}'."
        )

    cleared_at = _now_iso()
    
    # Remove any linked dynamic constraints
    linked = [c_id for c_id, c in list(DYNAMIC_CONSTRAINTS.items()) if c.get('linked_block_id') == element_id]
    for c_id in linked:
        del DYNAMIC_CONSTRAINTS[c_id]

    # Resume any trains that were halted solely because of this block
    resumed = []
    for t_id, state in TRAIN_STATES.items():
        if state.get("status") == "Halted":
            path = state.get("path", [])
            curr_edge = state.get("edge_id", "")
            # Check if the next edge (previously blocked) is now clear
            try:
                curr_idx = path.index(curr_edge)
                if curr_idx + 1 < len(path):
                    next_edge = path[curr_idx + 1]
                    if next_edge == element_id and element_id not in ACTIVE_BLOCKS:
                        state["status"] = "Moving"
                        resumed.append(t_id)
            except (ValueError, IndexError):
                pass
        elif state.get("status") == "Blocked" and state.get("edge_id") == element_id:
            # Train was stuck ON the now-cleared edge
            state["status"] = "Moving"
            resumed.append(t_id)

    # Re-sync RL env track_map (restored the cleared edge)
    _sync_blocks_to_rl_env()

    if resumed:
        print(f"[MMS] ▶️  Resumed {len(resumed)} train(s) after block cleared: {resumed}")

    AUDIT_LOGS.insert(0, {
        "t": cleared_at,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": f"MMS_{element_id}",
        "action": f"Maintenance cleared. {len(linked)} constraints lifted.",
        "operator": "Dispatcher",
        "status": "Cleared",
        "statusType": "success",
        "id": str(uuid.uuid4())
    })
    payload = {
        "type": "MAINTENANCE_CLEARED",
        "element_id": element_id,
        "cleared_at": cleared_at,
    }
    await _broadcast_topology(payload)
    await _broadcast_copilot(payload)

    print(f"[MMS] ✅ Block cleared on '{element_id}' at {cleared_at}")

    return {
        "status": "block_cleared",
        "element_id": element_id,
        "cleared_at": cleared_at,
    }


@app.post("/api/v1/simulation/analyze", tags=["Simulation Sandbox"])
async def analyze_simulation(req: WhatIfScenarioRequest):
    """
    Core two-stage simulation pipeline:
      1. RL Model  — proposes optimal adaptive actions for the current scenario.
      2. Operator forced_actions — override specific trains' proposed actions
         (models "what if I hold/divert this train" rather than only "what if delayed").
      3. OR-Tools  — validates and overrides any unsafe proposals (forced actions
         are NOT exempt from safety validation — an operator's hypothetical
         override can still be vetoed if it's unsafe, same as live override).
    Returns deterministic impact scores and human-readable network adjustments,
    tagged with the scenario label for client-side comparison.
    """
    delay_train_id = req.delay_train_id or ""
    latency_minutes = req.latency_minutes or 15
    forced_actions = req.forced_actions or {}

    # ── Step 1: Snapshot live network state ──────────────────────────────
    live_trains = list(TRAIN_STATES.values())
    n_live = len(live_trains)
    if n_live == 0:
        raise HTTPException(status_code=503, detail="No live train data available.")

    # ── Step 2: Inject delay into a working copy ──────────────────────────
    # We represent the network as an array the RL observation builder can consume.
    # Each train slot: [position_pct, priority, status_flag, delay_flag]
    from ai.config import MAX_TRAINS_CAPACITY, MAX_SPEED

    obs = np.zeros((MAX_TRAINS_CAPACITY, 10), dtype=np.float32)
    train_meta = []  # (train_id, edge_id, status, is_delayed)

    STATUS_SCORE = {"Moving": 0.8, "Waiting": 0.4, "Delayed": 0.2, "Scheduled": 0.1}
    PRIORITY_MAP_LOCAL = {
        "Vande Bharat": 10, "Rajdhani": 10, "Superfast": 8,
        "Express": 6, "Local": 5, "Suburban": 5,
        "Passenger": 3, "Freight (WAG-9)": 2,
    }

    for idx, t in enumerate(live_trains[:MAX_TRAINS_CAPACITY]):
        t_id      = t.get("train_id", "")
        pos_pct   = float(t.get("position_percentage", 0.0))
        status    = t.get("status", "Moving")
        is_delayed = (t_id == delay_train_id)

        # Derive priority from fleet registry or default
        fleet_cfg = FLEET_REGISTRY.get(t_id, {})
        train_type = fleet_cfg.get("train_type", "Express")
        priority = PRIORITY_MAP_LOCAL.get(train_type, 6)

        # Effective speed factor (degraded if delayed)
        speed_factor = 0.4 if is_delayed else STATUS_SCORE.get(status, 0.5)
        delay_flag   = min(latency_minutes, 60) / 60.0 if is_delayed else 0.0

        # Signal value: red (2.0 == danger) if delayed train blocks the segment
        signal_val = 2.0 if is_delayed else 0.0

        obs[idx] = [
            speed_factor,                   # normalised speed
            priority / 10.0,               # normalised priority
            signal_val / 2.0,              # signal (0=clear, 1=danger)
            0.5,                           # dist_to_danger (mid-range)
            0.5 - delay_flag * 0.3,        # dist_to_lead (shrinks under delay)
            speed_factor * 0.8,            # lead_speed proxy
            priority / 10.0,               # lead_priority proxy
            0.8,                           # dist_to_switch
            0.0,                           # dwell_rem
            max(0.0, 1.0 - delay_flag),    # deadline_rem (tighter under delay)
        ]
        train_meta.append((t_id, t.get("edge_id", "?"), status, is_delayed))

    # Pad remaining slots as ghost trains (already zeros)

    # ── Step 3: RL Model proposes actions ────────────────────────────────
    model, env = _get_sim_brain()
    if model is not None:
        # MaskablePPO expects obs shape (100, 10) — same as the env's observation_space
        proposed_actions, _ = model.predict(obs, deterministic=True)
        source = "RL+OR-Tools"
    else:
        # Fallback: greedy — every train wants to move (action=1)
        proposed_actions = np.ones(MAX_TRAINS_CAPACITY, dtype=np.int64)
        source = "OR-Tools only (model unavailable)"

    # ── Step 3b: Apply operator's hypothetical overrides ─────────────────
    # This is what makes the sandbox testable for "what if I hold/route this
    # train", not just "what if this train is late". Forced actions still
    # pass through the OR-Shield below — a hypothetical override that's
    # unsafe gets vetoed in the sandbox exactly like it would in production.
    forced_override_applied = []
    for idx, (t_id, edge_id, status, is_delayed) in enumerate(train_meta):
        if t_id in forced_actions and forced_actions[t_id] in (0, 1, 2):
            proposed_actions[idx] = forced_actions[t_id]
            forced_override_applied.append(t_id)

    # ── Step 4: OR-Tools safety validation ───────────────────────────────
    # Build a lightweight train list compatible with SmartOptimizer
    edge_to_pos = {}   # edge_id -> rough integer position
    for idx, (t_id, edge_id, status, _) in enumerate(train_meta):
        parts = edge_id.replace("edge-", "").split("-")
        try:
            pos = int(parts[0])
        except (ValueError, IndexError):
            pos = idx + 1
        edge_to_pos[t_id] = pos

    # Build a minimal track_map from the real NETWORK_TOPOLOGY edges
    topo_edges = NETWORK_TOPOLOGY.get("edges", [])
    topo_nodes = NETWORK_TOPOLOGY.get("nodes", [])
    node_ids   = [n["id"] for n in topo_nodes]
    track_map  = {}
    for edge in topo_edges:
        src_id = edge["source"]
        tgt_id = edge["target"]
        if src_id not in track_map:
            track_map[src_id] = {"next": [], "capacity": 2}
        track_map[src_id]["next"].append(tgt_id)

    or_trains = []
    for idx, (t_id, edge_id, status, is_delayed) in enumerate(train_meta):
        parts = edge_id.replace("edge-", "").split("-")
        try:
            src_node = parts[0]
        except IndexError:
            src_node = node_ids[0] if node_ids else "0"
        or_trains.append({
            "id":       t_id,
            "position": src_node,
            "speed":    50 if status == "Moving" else 0,
            "priority": PRIORITY_MAP_LOCAL.get(
                FLEET_REGISTRY.get(t_id, {}).get("train_type", "Express"), 6
            ),
        })

    safe_actions, _ = _OR_SHIELD.optimize_decision(
        or_trains,
        list(proposed_actions[:len(train_meta)]),
        track_map
    )

    # ── Step 5: Translate actions → human-readable adjustments ───────────
    ACTION_LABELS = {0: "HOLD", 1: "MAIN_LINE", 2: "DIVERT"}
    adjustments = []
    n_holds  = 0
    n_diverts = 0

    for idx, (t_id, edge_id, status, is_delayed) in enumerate(train_meta):
        ai_act   = int(proposed_actions[idx]) if idx < len(proposed_actions) else 1
        safe_act = int(safe_actions[idx])     if idx < len(safe_actions)     else ai_act
        vetoed   = (ai_act != 0 and safe_act == 0)  # OR-Tools overrode move → hold

        if safe_act == 0:
            n_holds += 1
            adj_type = "Signal Hold"
            if is_delayed:
                desc = (
                    f"{t_id} held at {edge_id} — primary delay source "
                    f"(+{latency_minutes} min). Adjacent trains cleared."
                )
            else:
                desc = (
                    f"{t_id} held at {edge_id} due to cascading delay from "
                    f"{delay_train_id}. OR-Shield veto: {'Yes' if vetoed else 'No'}."
                )
            adjustments.append({
                "id": len(adjustments)+1, 
                "type": adj_type, 
                "desc": desc,
                "train_id": t_id,
                "edge_id": edge_id,
                "constraint_type": "SPEED_LIMIT",
                "value": 0
            })

        elif safe_act == 2:
            n_diverts += 1
            adj_type = "Spatial Reroute"
            desc = f"{t_id} diverted off main line to loop to recover headway."
            adjustments.append({
                "id": len(adjustments)+1, 
                "type": adj_type, 
                "desc": desc,
                "train_id": t_id,
                "edge_id": edge_id,
                "constraint_type": "REROUTE",
                "value": 2
            })

    # Always add a speed-cap proposal for the delayed train itself
    if delay_train_id:
        speed_cap = max(20, 90 - latency_minutes * 2)
        delay_edge_id = next((e for t, e, _, _ in train_meta if t == delay_train_id), "edge-1-2")
        adjustments.insert(0, {
            "id": 0,
            "type": "Dynamic Speed Cap",
            "desc": (
                f"{delay_train_id} speed capped to {speed_cap} km/h to prevent "
                f"rear-end risk. Latency: +{latency_minutes} min."
            ),
            "train_id": delay_train_id,
            "edge_id": delay_edge_id,
            "constraint_type": "SPEED_LIMIT",
            "value": speed_cap
        })

    # ── Step 6: Compute real impact metrics ──────────────────────────────
    # Reliability degradation: proportional to delay severity + cascading holds
    base_reliability_hit = -(latency_minutes * 0.6 + n_holds * 2.5)
    reliability_pct = f"{max(base_reliability_hit, -95.0):.1f}%"

    # Congestion index: increases with latency and inversely with diversions
    congestion_base = 30 + latency_minutes * 1.8 + n_holds * 5 - n_diverts * 3
    congestion_pct  = f"+{min(congestion_base, 200):.0f}%"

    result = {
        "label": req.label,
        "source": source,
        "delay_train_id": delay_train_id,
        "latency_minutes": latency_minutes,
        "forced_actions_applied": forced_override_applied,
        "impact": {
            "reliability": reliability_pct,
            "congestion":  congestion_pct,
            "trains_held":    n_holds,
            "trains_diverted": n_diverts,
        },
        "adjustments": adjustments,
        "timestamp": _now_iso(),
    }
    print(f"[SIM-ANALYZE] ✅ Analysis complete | source={source} | holds={n_holds} diversions={n_diverts}")
    return result


@app.post("/api/v1/simulation/deploy", tags=["Simulation Sandbox"])
async def deploy_simulation(payload: dict):
    """
    Deploys a set of active blocks and dynamic constraints from the Sandbox
    into the live production environment.
    """
    blocks = payload.get("blocks", [])
    constraints = payload.get("constraints", [])

    # Apply constraints to DYNAMIC_CONSTRAINTS
    for c in constraints:
        c_id = c.get("id", f"constraint-{uuid.uuid4()}")
        DYNAMIC_CONSTRAINTS[c_id] = c
        
    # Apply What-If Blocks as Real Blocks
    for b in blocks:
        element_id = b.get("element_id")
        if not element_id:
            continue
            
        block_dict = {
            "blockId": b.get("blockId", str(uuid.uuid4())),
            "element_id": element_id,
            "type": b.get("type", "TRACK_FAULT"),
            "severity": b.get("severity", "TOTAL_BLOCK"),
            "reason": b.get("reason", "Deployed from Sandbox"),
            "applied_at": _now_iso(),
            "isWhatIf": False # Now it's real
        }
        
        ACTIVE_BLOCKS[element_id] = block_dict
        impact = _resolve_reroute_strategy(element_id)
        
        await _broadcast_topology({
            "type": "MAINTENANCE_BLOCK_APPLIED",
            "block": block_dict,
            "impact": impact,
        })
    
    # Audit Log
    AUDIT_LOGS.insert(0, {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SIMULATION_SANDBOX",
        "action": f"Deployed {len(blocks)} blocks and {len(constraints)} adjustments.",
        "operator": "Chief Dispatcher",
        "status": "Deployed",
        "statusType": "success",
        "id": str(uuid.uuid4())
    })

    print(f"[SANDBOX] 🚀 Simulation Deployed! ({len(blocks)} blocks, {len(constraints)} constraints)")
    return {"status": "success", "blocks_applied": len(blocks), "constraints_applied": len(constraints)}


@app.get("/api/v1/impact-analysis", tags=["MMS"])
async def get_impact_analysis():
    """
    Return a consolidated impact report for ALL currently active blocks.
    Used by the frontend to trigger the 'Ripple Effect' notification.
    """
    if not ACTIVE_BLOCKS:
        return {
            "status": "clear",
            "message": "No active maintenance blocks. Network operating normally.",
            "total_affected_trains": 0,
            "blocks": [],
            "timestamp": _now_iso(),
        }

    reports = []
    total_affected: set = set()

    for element_id in ACTIVE_BLOCKS:
        report = _resolve_reroute_strategy(element_id)
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
            f"Maintenance on {len(ACTIVE_BLOCKS)} segment(s) affects {n} upcoming train(s). "
            f"Rerouting strategy: {strategy_str}."
        ),
        "block_reports": reports,
        "timestamp": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Fleet Management — seed FLEET_REGISTRY from initial TRAIN_STATES on startup
# ---------------------------------------------------------------------------
def _seed_fleet_registry():
    """
    Build FLEET_REGISTRY from the initial TRAIN_STATES simulation data.
    Called once from startup_event after simulate_trains_bg has initialised.
    """
    # Speed limits per train type (km/h)
    MAX_SPEED_MAP = {
        "Vande Bharat": 130, "Rajdhani": 120, "Superfast": 110,
        "Express": 90, "Local": 75, "Suburban": 75,
        "Passenger": 60, "Freight (WAG-9)": 40,
    }

    for t_id, state in TRAIN_STATES.items():
        # Read train_type directly from TRAIN_STATES if available
        train_type = state.get("train_type", "Express")
        max_speed  = state.get("speed_kmh") or MAX_SPEED_MAP.get(train_type, 90)
        
        # Preserve priority if it already exists in state; otherwise try to map it
        priority   = state.get("priority")
        if priority is None:
            priority = PRIORITY_MAP.get(train_type.title(), 6)

        # Preserve configuration if it exists, otherwise pull from TRAIN_STATES
        existing_cfg = FLEET_REGISTRY.get(t_id, {})
        direction    = existing_cfg.get("direction") or state.get("direction", "DOWN")
        start_time   = existing_cfg.get("start_time") or state.get("start_time", 0)
        deadline     = existing_cfg.get("deadline") or state.get("deadline", 120)

        # Always sync — ensures type/priority/speed stay correct after restarts
        FLEET_REGISTRY[t_id] = {
            "train_id"  : t_id,
            "train_type": train_type,
            "max_speed" : max_speed,
            "priority"  : priority,
            "start_time": start_time,
            "deadline"  : deadline,
            "direction" : direction,
            "path"      : state.get("path", []),
            "added_at"  : existing_cfg.get("added_at", _now_iso()),
        }


@app.on_event("startup")
async def _fleet_seed_on_startup():
    """Seed the fleet registry once the simulation loop has populated TRAIN_STATES."""
    await asyncio.sleep(1)   # give simulate_trains_bg time to initialise
    _seed_fleet_registry()


# ---------------------------------------------------------------------------
# Fleet Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/fleet", tags=["Fleet"])
async def get_fleet():
    """
    Return live fleet: merges FLEET_REGISTRY config with real-time TRAIN_STATES.
    Always reflects current edge, position, and status from the simulation.
    """
    _seed_fleet_registry()   # ensure any newly spawned trains are included

    result = []
    for t_id, cfg in FLEET_REGISTRY.items():
        live = TRAIN_STATES.get(t_id, {})
        result.append({
            **cfg,
            "edge_id"            : live.get("edge_id", "—"),
            "position_percentage": live.get("position_percentage", 0),
            "status"             : live.get("status", "Scheduled"),
            "speed_kmh"          : live.get("speed_kmh", 0),
        })

    # Also include any live trains not yet in FLEET_REGISTRY
    for t_id, live in TRAIN_STATES.items():
        if t_id not in FLEET_REGISTRY:
            result.append({
                "train_id"           : t_id,
                "train_type"         : "Express",
                "max_speed"          : 110,
                "priority"           : 6,
                "start_time"         : 0,
                "deadline"           : 120,
                "direction"          : live.get("direction", "UP" if "UP" in t_id else "DOWN"),
                "path"               : live.get("path", []),
                "added_at"           : _now_iso(),
                "edge_id"            : live.get("edge_id", "—"),
                "position_percentage": live.get("position_percentage", 0),
                "status"             : live.get("status", "Moving"),
                "speed_kmh"          : live.get("speed_kmh", 0),
            })

    return {"fleet": result, "count": len(result), "timestamp": _now_iso()}


@app.post("/api/v1/fleet", tags=["Fleet"])
async def add_train(req: NewTrainRequest):
    """
    Add a new train to the session fleet.
    - Validates train_id uniqueness.
    - Derives priority from type if not supplied.
    - Seeds both FLEET_REGISTRY and TRAIN_STATES so the map shows it immediately.
    - Broadcasts topology_update so all map clients see the new train.
    """
    if req.train_id in FLEET_REGISTRY or req.train_id in TRAIN_STATES:
        raise HTTPException(
            status_code=409,
            detail=f"Train '{req.train_id}' already exists in the fleet."
        )

    if req.train_type not in TRAIN_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid train_type '{req.train_type}'. Must be one of: {TRAIN_TYPES}"
        )

    # Derive priority from type
    priority = req.priority if req.priority is not None else PRIORITY_MAP.get(req.train_type, 5)

    # Build path from topology (first edge → destination)
    all_edges = [e["id"] for e in NETWORK_TOPOLOGY.get("edges", [])]
    default_path = all_edges[:8] if len(all_edges) >= 8 else all_edges  # first 8 edges as path

    dir_str = "UP" if req.direction == 1 else "DOWN"
    cfg = {
        "train_id"  : req.train_id,
        "train_type": req.train_type,
        "max_speed" : req.max_speed,
        "priority"  : priority,
        "start_time": req.start_time,
        "deadline"  : req.deadline,
        "direction" : dir_str,
        "path"      : default_path,
        "added_at"  : _now_iso(),
    }
    FLEET_REGISTRY[req.train_id] = cfg

    # Seed live simulation state so train appears on the map
    TRAIN_STATES[req.train_id] = {
        "train_id"           : req.train_id,
        "edge_id"            : default_path[0] if default_path else "edge-0-1",
        "position_percentage": 0.0,
        "status"             : "Moving",
        "path"               : default_path,
        "direction"          : dir_str,
    }

    # Broadcast updated train list to all topology WS clients
    await _broadcast_topology({
        "type"  : "topology_update",
        "trains": list(TRAIN_STATES.values()),
        "conflicts": [],
        "maintenance_blocks": list(ACTIVE_BLOCKS.values()),
    })

    print(f"[Fleet] ➕ Added train '{req.train_id}' (type={req.train_type}, prio={priority})")
    return {"status": "added", "train": cfg, "timestamp": cfg["added_at"]}


def _get_rerouted_path(original_path: List[str]) -> List[str]:
    """
    Scans the path for blocked segments. If a block is found on any junction
    edge (switch node with >1 next options), finds a free sibling platform/loop.
    Fully dynamic — works for any topology without hardcoded node IDs.
    """
    new_path = []
    i = 0
    while i < len(original_path):
        edge_id = original_path[i]
        if edge_id in ACTIVE_BLOCKS:
            parts = edge_id.split("-")
            if len(parts) >= 3:
                try:
                    src = int(parts[1])
                except ValueError:
                    new_path.append(edge_id)
                    i += 1
                    continue

                # Dynamic junction detection: any node with multiple exits is a switch
                src_nexts = RAW_TRACK_MAP.get(src, {}).get("next", [])
                if len(src_nexts) > 1:
                    found_bypass = False
                    for alt_dst in src_nexts:
                        alt_edge_in = f"edge-{src}-{alt_dst}"
                        if alt_edge_in == edge_id or alt_edge_in in ACTIVE_BLOCKS:
                            continue
                        # Check the exit from the alternate sibling is also clear
                        exits = RAW_TRACK_MAP.get(alt_dst, {}).get("next", [])
                        if exits:
                            alt_edge_out = f"edge-{alt_dst}-{exits[0]}"
                            if alt_edge_out not in ACTIVE_BLOCKS:
                                # SUCCESS: replace blocked entry+exit with bypass pair
                                new_path.append(alt_edge_in)
                                new_path.append(alt_edge_out)
                                i += 2  # skip the blocked edge AND its exit
                                found_bypass = True
                                break
                    if found_bypass:
                        continue

        new_path.append(edge_id)
        i += 1
    return new_path

@app.post("/api/v1/fleet/generate-schedule", tags=["Fleet"])
async def generate_schedule():
    """
    Run OR-Tools on the current FLEET_REGISTRY to produce a conflict-free
    base schedule for this session.

    Stores the result in LAST_OR_SCHEDULE so /start-inference can consume
    it to seed TRAIN_STATES and kick off the RL inference loop.
    """
    from or_tools.corridor_planner import CorridorPlanner
    from ai.map_generator import STATIONS, generate_realistic_section
    from ai.config import generate_daily_schedule, ARCHETYPE_BY_NAME

    if len(FLEET_REGISTRY) < 25:
        # Auto-pad missing trains instead of breaking the model
        needed = 25 - len(FLEET_REGISTRY)
        fleet, schedule_map = generate_daily_schedule(num_trains=needed)
        for t in fleet:
            t_sched = schedule_map.get(t['id'], {})
            
            base_id = t['id']
            unique_id = base_id
            counter = 1
            while unique_id in FLEET_REGISTRY:
                unique_id = f"{base_id}-{counter}"
                counter += 1
                
            t['id'] = unique_id
            t['train_id'] = unique_id
            t['path'] = []
            t['train_type'] = t.get('archetype', 'Express')
            t['start_time'] = t_sched.get('start_time', 0)
            t['deadline'] = t_sched.get('deadline', 100)
            
            dir_str = "UP" if "UP" in unique_id else "DOWN"
            t['direction'] = t.get('direction', dir_str)
            
            FLEET_REGISTRY[t['id']] = t
    elif not FLEET_REGISTRY:
        fleet, schedule_map = generate_daily_schedule(num_trains=25)
        for t in fleet:
            # Map Python simulator archetype/schedule back to the API/Frontend schema
            t_sched = schedule_map.get(t['id'], {})
            t['path'] = []
            t['train_id'] = t['id']
            t['train_type'] = t.get('archetype', 'Express')
            t['start_time'] = t_sched.get('start_time', 0)
            t['deadline'] = t_sched.get('deadline', 100)
            
            FLEET_REGISTRY[t['id']] = t

    topo_nodes = {n["id"]: n for n in NETWORK_TOPOLOGY.get("nodes", [])}
    track_map  = {}
    for nid, ndata in topo_nodes.items():
        track_map[nid] = {
            "type"    : ndata.get("type", "BLOCK"),
            "capacity": 2 if ndata.get("type") in ("PLATFORM", "LOOP") else 1,
            "next"    : ndata.get("next", [])
        }

    _, _, _, _, token_blocks = generate_realistic_section()
    planner = CorridorPlanner(track_map, STATIONS, token_blocks)

    active_fleet = []
    schedule_req = {}

    for t_id, cfg in FLEET_REGISTRY.items():
        # DYNAMIC REROUTING: Check for maintenance blocks and pick alternative platforms/loops
        original_path = cfg.get("path", [])
        if original_path:
            path = _get_rerouted_path(original_path)
            cfg["path"] = path # update the registry so start-inference uses the new path

        train_type = cfg.get("train_type", "Express")
        direction_val = cfg.get("direction", 1)
        direction_str = "DOWN" if direction_val in (1, "DOWN") else "UP"

        train_type_upper = train_type.upper()
        if "EXPRESS" in train_type_upper or "MAIL" in train_type_upper:
            train_type_upper = "MAIL_EXPRESS"
        elif "FREIGHT" in train_type_upper or "GOODS" in train_type_upper:
            train_type_upper = "GOODS"
        elif "LOCAL" in train_type_upper or "SUBURBAN" in train_type_upper:
            train_type_upper = "PASSENGER"
        elif "VANDE BHARAT" in train_type_upper:
            train_type_upper = "RAJDHANI"

        # Lookup archetype for stops and banker requirement
        archetype = ARCHETYPE_BY_NAME.get(train_type_upper, ARCHETYPE_BY_NAME["MAIL_EXPRESS"])
        stops = archetype.get("stops_down" if direction_str == "DOWN" else "stops_up", [])

        active_fleet.append({
            "id": t_id,
            "direction": direction_str,
            "priority": cfg.get("priority", 5),
            "max_speed": cfg.get("max_speed", 100),
            "banker_required": archetype.get("banker_required", False),
            "finished": False,
            "position": 0 if direction_str == "DOWN" else 998, # Start position so it's not finished
        })
        schedule_req[t_id] = {
            "stops": stops,
            "start_time": cfg.get("start_time", 0),
            "deadline": cfg.get("deadline", 120),
        }

    if not active_fleet:
        raise HTTPException(status_code=400, detail="No valid trains to schedule.")

    result = planner.solve(active_fleet, schedule_req, sim_time=0)

    if result is None:
        return {
            "status"    : "infeasible",
            "message"   : "OR-Tools could not find a feasible schedule within the time horizon.",
            "fleet_size": len(active_fleet),
            "timestamp" : _now_iso(),
        }

    # Store schedule so start-inference can consume it
    global LAST_OR_SCHEDULE
    LAST_OR_SCHEDULE = result.get("schedule", {})
    print(f"[ORBIT] 📅 OR Schedule stored for {len(LAST_OR_SCHEDULE)} trains.")

    return {
        "status"        : "optimal",
        "fleet_size"    : len(active_fleet),
        "schedule"      : result.get("schedule", {}),
        "expert_actions": result.get("expert_actions", {}),
        "timestamp"     : _now_iso(),
    }
